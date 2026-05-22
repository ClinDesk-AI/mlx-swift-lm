// Copyright © 2026 ClinDesk LLC.

import Foundation
import MLX
import MLXNN

private final class Gemma4AudioRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

private final class Gemma4AudioClippableLinear: Module, UnaryLayer {
    let useClipping: Bool

    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray?
    @ModuleInfo(key: "input_max") var inputMax: MLXArray?
    @ModuleInfo(key: "output_min") var outputMin: MLXArray?
    @ModuleInfo(key: "output_max") var outputMax: MLXArray?

    init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool = true) {
        self.useClipping = useClipping
        self._linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)
        if useClipping {
            self._inputMin.wrappedValue = MLXArray(-Float.infinity)
            self._inputMax.wrappedValue = MLXArray(Float.infinity)
            self._outputMin.wrappedValue = MLXArray(-Float.infinity)
            self._outputMax.wrappedValue = MLXArray(Float.infinity)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let clippedInput =
            if let inputMin, let inputMax {
                clip(x, min: inputMin, max: inputMax)
            } else {
                x
            }
        let projected = linear(clippedInput)
        if let outputMin, let outputMax {
            return clip(projected, min: outputMin, max: outputMax)
        }
        return projected
    }
}

private final class Gemma4SSCPConvBlock: Module {
    private let timeStride = 2
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(config: Gemma4AudioConfiguration, index: Int) {
        let inputChannels = index == 0 ? 1 : config.subsamplingConvChannels[index - 1]
        let outputChannels = config.subsamplingConvChannels[index]
        self._conv.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: [3, 3],
            stride: [2, 2],
            padding: 0,
            bias: false
        )
        self._norm.wrappedValue = LayerNorm(
            dimensions: outputChannels,
            eps: config.rmsNormEps,
            affine: true,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var x = MLX.where(
            mask[0..., 0..., .newAxis, .newAxis],
            MLXArray(0.0, dtype: x.dtype),
            x
        )
        x = padded(x, widths: [0, [1, 1], [1, 1], 0])
        x = conv(x)

        let outputLength = x.dim(1)
        let outputMask = mask[0..., .stride(by: timeStride)][0..., ..<outputLength]
        x = relu(norm(x))
        return (x, outputMask)
    }
}

private final class Gemma4SubSampleConvProjection: Module {
    static let inputFeatureSize = 128

    @ModuleInfo(key: "layer0") var layer0: Gemma4SSCPConvBlock
    @ModuleInfo(key: "layer1") var layer1: Gemma4SSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjectionLinear: Linear

    init(config: Gemma4AudioConfiguration) {
        self._layer0.wrappedValue = Gemma4SSCPConvBlock(config: config, index: 0)
        self._layer1.wrappedValue = Gemma4SSCPConvBlock(config: config, index: 1)

        var frequency = Self.inputFeatureSize
        for _ in 0 ..< 2 {
            frequency = (frequency + 2 - 3) / 2 + 1
        }
        let projectedInputSize = frequency * config.subsamplingConvChannels.last!
        self._inputProjectionLinear.wrappedValue = Linear(
            projectedInputSize,
            config.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ audioMel: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var x = expandedDimensions(audioMel, axis: -1)
        var mask = mask

        (x, mask) = layer0(x, mask: mask)
        (x, mask) = layer1(x, mask: mask)

        let batch = x.dim(0)
        let time = x.dim(1)
        let frequency = x.dim(2)
        let channels = x.dim(3)
        x = x.reshaped(batch, time, frequency * channels)
        x = inputProjectionLinear(x)
        return (x, mask)
    }
}

private final class Gemma4ConformerFeedForward: Module, UnaryLayer {
    let gradientClipping: Float
    let residualWeight: Float

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: Gemma4AudioClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: Gemma4AudioClippableLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: Gemma4AudioRMSNorm

    init(config: Gemma4AudioConfiguration) {
        self.gradientClipping = config.gradientClipping
        self.residualWeight = config.residualWeight
        self._preLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._ffwLayer1.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.hiddenSize * 4,
            bias: false,
            useClipping: config.useClippedLinears
        )
        self._ffwLayer2.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize * 4,
            outFeatures: config.hiddenSize,
            bias: false,
            useClipping: config.useClippedLinears
        )
        self._postLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var x = clip(x, min: -gradientClipping, max: gradientClipping)
        x = preLayerNorm(x)
        x = ffwLayer1(x)
        x = silu(x)
        x = ffwLayer2(x)
        x = clip(x, min: -gradientClipping, max: gradientClipping)
        x = postLayerNorm(x)
        return residual + x * residualWeight
    }
}

private final class Gemma4AudioRelativePositionEmbedding {
    let numHeads: Int
    let channels: Int
    let headDim: Int
    let maxBackward: Int
    let maxForward: Int
    let invTimescales: MLXArray

    let positionProjection: Linear

    init(config: Gemma4AudioConfiguration) {
        self.numHeads = config.attentionHeads
        self.channels = config.hiddenSize
        self.headDim = config.hiddenSize / config.attentionHeads
        self.maxBackward = max(0, config.attentionContextLeft - 1)
        self.maxForward = config.attentionContextRight
        self.positionProjection = Linear(
            config.hiddenSize,
            config.attentionHeads * (config.hiddenSize / config.attentionHeads),
            bias: false
        )

        let timescales = config.hiddenSize / 2
        let increment = log(10_000.0) / Float(max(timescales - 1, 1))
        self.invTimescales = exp(arange(timescales).asType(.float32) * -increment)
            .reshaped(1, 1, timescales)
    }

    init(config: Gemma4AudioConfiguration, projection: Linear) {
        self.numHeads = config.attentionHeads
        self.channels = config.hiddenSize
        self.headDim = config.hiddenSize / config.attentionHeads
        self.maxBackward = max(0, config.attentionContextLeft - 1)
        self.maxForward = config.attentionContextRight
        self.positionProjection = projection

        let timescales = config.hiddenSize / 2
        let increment = log(10_000.0) / Float(max(timescales - 1, 1))
        self.invTimescales = exp(arange(timescales).asType(.float32) * -increment)
            .reshaped(1, 1, timescales)
    }

    private func timingSignal(position: MLXArray, dtype: DType) -> MLXArray {
        let scaledTime = position.asType(.float32)[.ellipsis, .newAxis] * invTimescales
        let signal = concatenated([sin(scaledTime), cos(scaledTime)], axis: -1)
        return signal.asType(dtype)
    }

    private func relativeShift(
        _ termBD: MLXArray,
        batchSize: Int,
        numHeads: Int,
        numBlocks: Int,
        blockSize: Int,
        contextSize: Int,
        maxSpanPlusOne: Int
    ) -> MLXArray {
        let padAmount = (contextSize + 1) - maxSpanPlusOne
        var termBD = padded(termBD, widths: [0, 0, 0, 0, [0, padAmount]])
        termBD = termBD.reshaped(batchSize, numHeads, numBlocks, blockSize * (contextSize + 1))
        termBD = termBD[0..., 0..., 0..., ..<(blockSize * contextSize)]
        termBD = termBD.reshaped(batchSize, numHeads, numBlocks, blockSize, contextSize)
        return termBD
    }

    func callAsFunction(_ queries: MLXArray, keys: MLXArray) -> MLXArray {
        let batch = queries.dim(0)
        let numBlocks = queries.dim(1)
        let blockSize = queries.dim(2)
        let heads = queries.dim(3)
        let headDim = queries.dim(4)
        let contextSize = keys.dim(2)

        let positions = arange(maxBackward, -maxForward - 1, step: -1)[.newAxis, 0...]
        let maxSpanPlusOne = positions.dim(1)

        var sinusoidal = timingSignal(position: positions, dtype: queries.dtype)
        sinusoidal = positionProjection(sinusoidal.asType(positionProjection.weight.dtype))
        sinusoidal = sinusoidal.reshaped(maxSpanPlusOne, numHeads, self.headDim)
            .asType(queries.dtype)

        let queriesProjected = queries.transposed(0, 3, 1, 2, 4)
        let keysProjected = keys.transposed(0, 3, 1, 4, 2)
        let termAC = matmul(queriesProjected, keysProjected)

        let sinusoidalTransposed = sinusoidal.transposed(1, 2, 0)
        let reshapedQueries = queriesProjected.reshaped(batch, heads, numBlocks * blockSize, headDim)
        let termBD = matmul(reshapedQueries, sinusoidalTransposed)
            .reshaped(batch, heads, numBlocks, blockSize, maxSpanPlusOne)

        return termAC + relativeShift(
            termBD,
            batchSize: batch,
            numHeads: heads,
            numBlocks: numBlocks,
            blockSize: blockSize,
            contextSize: contextSize,
            maxSpanPlusOne: maxSpanPlusOne
        )
    }
}

private final class Gemma4AudioAttention: Module {
    let numHeads: Int
    let hiddenSize: Int
    let headDim: Int
    let chunkSize: Int
    let maxFutureHorizon: Int
    let maxPastHorizon: Int
    let contextSize: Int
    let invalidLogitsValue: Float
    let softcap: Float
    let qScale: Float
    let kScale: Float

    @ModuleInfo(key: "relative_k_proj") var relativeKeyProjection: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray
    @ModuleInfo(key: "q_proj") var qProjection: Gemma4AudioClippableLinear
    @ModuleInfo(key: "k_proj") var kProjection: Gemma4AudioClippableLinear
    @ModuleInfo(key: "v_proj") var vProjection: Gemma4AudioClippableLinear
    @ModuleInfo(key: "post") var postProjection: Gemma4AudioClippableLinear

    private let relativePosition: Gemma4AudioRelativePositionEmbedding

    init(config: Gemma4AudioConfiguration) {
        self.numHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize
        self.headDim = config.hiddenSize / config.attentionHeads
        self.chunkSize = config.attentionChunkSize
        self.maxFutureHorizon = config.attentionContextRight
        self.maxPastHorizon = max(0, config.attentionContextLeft - 1)
        self.contextSize = chunkSize + maxPastHorizon + maxFutureHorizon
        self.invalidLogitsValue = config.attentionInvalidLogitsValue
        self.softcap = config.attentionLogitCap
        self.qScale = pow(Float(headDim), -0.5) / log(2.0)
        self.kScale = log(1.0 + Float(M_E)) / log(2.0)

        self._relativeKeyProjection.wrappedValue = Linear(
            hiddenSize,
            numHeads * headDim,
            bias: false
        )
        self._perDimScale.wrappedValue = MLXArray.zeros([headDim])
        self._qProjection.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numHeads * headDim,
            bias: false,
            useClipping: config.useClippedLinears
        )
        self._kProjection.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numHeads * headDim,
            bias: false,
            useClipping: config.useClippedLinears
        )
        self._vProjection.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: numHeads * headDim,
            bias: false,
            useClipping: config.useClippedLinears
        )
        self._postProjection.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: hiddenSize,
            outFeatures: hiddenSize,
            bias: false,
            useClipping: config.useClippedLinears
        )
        self.relativePosition = Gemma4AudioRelativePositionEmbedding(
            config: config,
            projection: self._relativeKeyProjection.wrappedValue
        )
        super.init()
    }

    private func padSequence(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        var widths = Array(repeating: IntOrPair(0), count: x.ndim)
        widths[1] = [left, right]
        return padded(x, widths: widths)
    }

    private func convertToBlock(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let rest = Array(x.shape.dropFirst(2))
        let blocks = (time + chunkSize - 1) / chunkSize
        let padLength = blocks * chunkSize - time
        let paddedInput = padLength > 0 ? padSequence(x, left: 0, right: padLength) : x
        return paddedInput.reshaped([batch, blocks, chunkSize] + rest)
    }

    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let paddedInput = padSequence(
            x,
            left: maxPastHorizon,
            right: maxFutureHorizon + chunkSize - 1
        )
        let paddedTime = paddedInput.dim(1)
        let blocks = (paddedTime - contextSize) / chunkSize + 1
        let starts = arange(blocks) * chunkSize
        let offsets = arange(contextSize)
        let indices = starts[0..., .newAxis] + offsets[.newAxis, 0...]

        switch paddedInput.ndim {
        case 2:
            return paddedInput[0..., indices]
        case 3:
            return paddedInput[0..., indices, 0...]
        case 4:
            return paddedInput[0..., indices, 0..., 0...]
        default:
            fatalError("Unsupported audio block context rank \(paddedInput.ndim)")
        }
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray,
        causalValidMask: MLXArray
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let time = hiddenStates.dim(1)
        let qkvShape = [batch, time, numHeads, headDim]

        var q = qProjection(hiddenStates).asType(.float32).reshaped(qkvShape)
        var k = kProjection(hiddenStates).asType(.float32).reshaped(qkvShape)
        let v = vProjection(hiddenStates).asType(.float32).reshaped(qkvShape)

        let dimScale = softplus(perDimScale)
        q = q * (qScale * dimScale)
        k = k * kScale

        let queryBlocks = convertToBlock(q)
        let keyBlocks = extractBlockContext(k)
        let valueBlocks = extractBlockContext(v)
        let blockCount = queryBlocks.dim(1)

        let validMask = logicalNot(mask)
        let extractedValid = extractBlockContext(validMask)
        let condition = logicalAnd(
            extractedValid[0..., .newAxis, 0..., .newAxis, 0...],
            causalValidMask[.newAxis, .newAxis, .newAxis, 0..., 0...]
        )

        var logits = relativePosition(queryBlocks, keys: keyBlocks)
        logits = tanh(logits / softcap) * softcap
        logits = MLX.where(
            condition,
            logits,
            MLXArray(invalidLogitsValue, dtype: logits.dtype)
        )

        let probabilities = softmax(logits, axis: -1)
        var context = einsum("bnuwc,bucnh->buwnh", probabilities, valueBlocks)
        context = context.reshaped(batch, blockCount * chunkSize, numHeads, headDim)
        context = context[0..., ..<time, 0..., 0...]

        context = context.reshaped(batch, time, numHeads * headDim)
        return postProjection(context)
    }
}

private final class Gemma4ConformerLightConv1d: Module, UnaryLayer {
    let gradientClipping: Float
    let causalPadding: Int

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: Gemma4AudioClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: Gemma4AudioClippableLinear

    init(config: Gemma4AudioConfiguration) {
        self.gradientClipping = config.gradientClipping
        self.causalPadding = config.convKernelSize - 1
        self._preLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._linearStart.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.hiddenSize * 2,
            bias: false,
            useClipping: config.useClippedLinears
        )
        self._depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize,
            stride: 1,
            padding: 0,
            groups: config.hiddenSize,
            bias: false
        )
        self._convNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._linearEnd.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.hiddenSize,
            bias: false,
            useClipping: config.useClippedLinears
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var x = preLayerNorm(x)
        x = linearStart(x)
        let parts = split(x, parts: 2, axis: -1)
        x = parts[0] * sigmoid(parts[1])
        x = padded(x, widths: [0, [causalPadding, 0], 0])
        x = depthwiseConv1d(x)
        x = clip(x, min: -gradientClipping, max: gradientClipping)
        x = convNorm(x)
        x = silu(x)
        x = linearEnd(x)
        return x + residual
    }
}

private final class Gemma4ConformerBlock: Module {
    let gradientClipping: Float

    @ModuleInfo(key: "feed_forward1") var feedForward1: Gemma4ConformerFeedForward
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4AudioAttention
    @ModuleInfo(key: "lconv1d") var lightConv1d: Gemma4ConformerLightConv1d
    @ModuleInfo(key: "feed_forward2") var feedForward2: Gemma4ConformerFeedForward
    @ModuleInfo(key: "norm_pre_attn") var normPreAttention: Gemma4AudioRMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttention: Gemma4AudioRMSNorm
    @ModuleInfo(key: "norm_out") var normOut: Gemma4AudioRMSNorm

    init(config: Gemma4AudioConfiguration) {
        self.gradientClipping = config.gradientClipping
        self._feedForward1.wrappedValue = Gemma4ConformerFeedForward(config: config)
        self._selfAttention.wrappedValue = Gemma4AudioAttention(config: config)
        self._lightConv1d.wrappedValue = Gemma4ConformerLightConv1d(config: config)
        self._feedForward2.wrappedValue = Gemma4ConformerFeedForward(config: config)
        self._normPreAttention.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._normPostAttention.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._normOut.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> MLXArray {
        var x = feedForward1(x)

        let residual = x
        x = clip(x, min: -gradientClipping, max: gradientClipping)
        x = normPreAttention(x)
        x = selfAttention(x, mask: mask, causalValidMask: causalValidMask)
        x = clip(x, min: -gradientClipping, max: gradientClipping)
        x = residual + normPostAttention(x)

        let validityMask = logicalNot(mask)[0..., 0..., .newAxis].asType(x.dtype)
        x = x * validityMask

        x = lightConv1d(x)
        x = feedForward2(x)
        x = clip(x, min: -gradientClipping, max: gradientClipping)
        return normOut(x)
    }
}

final class Gemma4AudioEncoder: Module {
    let config: Gemma4AudioConfiguration

    @ModuleInfo(key: "subsample_conv_projection") fileprivate var subsampleConvProjection:
        Gemma4SubSampleConvProjection
    @ModuleInfo(key: "layers") fileprivate var layers: [Gemma4ConformerBlock]
    @ModuleInfo(key: "output_proj") var outputProjection: Linear?

    init(config: Gemma4AudioConfiguration) {
        self.config = config
        self._subsampleConvProjection.wrappedValue = Gemma4SubSampleConvProjection(config: config)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4ConformerBlock(config: config)
        }
        if let outputProjectionDimensions = config.outputProjectionDimensions {
            self._outputProjection.wrappedValue = Linear(
                config.hiddenSize,
                outputProjectionDimensions,
                bias: true
            )
        }
        super.init()
    }

    private func causalValidMask() -> MLXArray {
        let chunkSize = config.attentionChunkSize
        let maxFutureHorizon = config.attentionContextRight
        let maxPastHorizon = max(0, config.attentionContextLeft - 1)
        let upperDiagonal = maxPastHorizon + maxFutureHorizon
        let contextSize = chunkSize + maxPastHorizon + maxFutureHorizon

        let lowerCausal = tril(MLXArray.ones([contextSize, chunkSize])).transposed()
        let upperCausal = tril(MLXArray.ones([chunkSize, contextSize]), k: upperDiagonal)
        return (lowerCausal * upperCausal).asType(.bool)
    }

    func callAsFunction(_ audioMel: MLXArray, mask audioMelMask: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        var (audioEncodings, currentMask) = subsampleConvProjection(audioMel, mask: audioMelMask)
        let causalValidMask = causalValidMask()

        for layer in layers {
            audioEncodings = layer(audioEncodings, mask: currentMask, causalValidMask: causalValidMask)
        }

        if let outputProjection {
            audioEncodings = outputProjection(audioEncodings)
        }

        if currentMask.dim(1) != audioEncodings.dim(1) {
            currentMask = currentMask[0..., ..<audioEncodings.dim(1)]
        }

        audioEncodings = MLX.where(
            currentMask[0..., 0..., .newAxis],
            MLXArray(0.0, dtype: audioEncodings.dtype),
            audioEncodings
        )

        return (audioEncodings, currentMask)
    }
}
