// Copyright © 2026 ClinDesk LLC.

@preconcurrency import AVFoundation
import Foundation
import MLX
import MLXLMCommon

private final class Gemma4AudioConverterInputState: @unchecked Sendable {
    var didProvideInput = false
}

public struct Gemma4PreparedAudio {
    public let features: MLXArray
    public let mask: MLXArray
    public let durationMilliseconds: Double
}

public struct Gemma4AudioFeatureExtractor {
    public let featureSize: Int
    public let samplingRate: Int
    public let frameLength: Int
    public let hopLength: Int
    public let fftLength: Int
    public let maxLength: Int
    public let padMultiple: Int
    public let melFloor: Float

    private let window: [Float]
    private let melFilters: MLXArray

    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16_000,
        frameLengthMilliseconds: Double = 20,
        hopLengthMilliseconds: Double = 10,
        maxLength: Int = 480_000,
        padMultiple: Int = 128,
        melFloor: Float = 1e-3
    ) {
        let computedFrameLength = Int(round(Double(samplingRate) * frameLengthMilliseconds / 1000.0))
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.frameLength = computedFrameLength
        self.hopLength = Int(round(Double(samplingRate) * hopLengthMilliseconds / 1000.0))
        self.fftLength = 1 << Int(ceil(log2(Double(computedFrameLength))))
        self.maxLength = maxLength
        self.padMultiple = padMultiple
        self.melFloor = melFloor

        self.window = (0 ..< computedFrameLength).map { index in
            0.5 - 0.5 * cos(2.0 * .pi * Float(index) / Float(computedFrameLength))
        }
        self.melFilters = Self.makeMelFilterBank(
            frequencyBins: self.fftLength / 2 + 1,
            melFilters: featureSize,
            minFrequency: 0,
            maxFrequency: Float(samplingRate) / 2,
            samplingRate: samplingRate
        )
    }

    public func extract(audio: UserInput.Audio) throws -> Gemma4PreparedAudio {
        switch audio {
        case .samples(let samples, let sampleRate):
            return extract(samples: resampleIfNeeded(samples, from: sampleRate))
        case .url(let url):
            return try extract(samples: Self.loadMonoSamples(url: url, sampleRate: samplingRate))
        }
    }

    public func extract(samples rawSamples: [Float]) -> Gemma4PreparedAudio {
        let trimmedSamples = Array(rawSamples.prefix(maxLength))
        let durationMilliseconds = Double(trimmedSamples.count) / Double(samplingRate) * 1000

        var targetLength = min(trimmedSamples.count, maxLength)
        if targetLength % padMultiple != 0 {
            targetLength = ((targetLength / padMultiple) + 1) * padMultiple
        }

        var samples = Array(repeating: Float(0), count: targetLength)
        var sampleMask = Array(repeating: false, count: targetLength)
        if !trimmedSamples.isEmpty {
            samples.replaceSubrange(0 ..< trimmedSamples.count, with: trimmedSamples)
            sampleMask.replaceSubrange(
                0 ..< trimmedSamples.count,
                with: Array(repeating: true, count: trimmedSamples.count)
            )
        }

        let leftPadding = frameLength / 2
        let waveform = Array(repeating: Float(0), count: leftPadding) + samples
        let attentionMask = Array(repeating: false, count: leftPadding) + sampleMask
        let frameSizeForUnfold = frameLength + 1
        let frameCount = max((waveform.count - frameSizeForUnfold) / hopLength + 1, 0)

        var framed = Array(repeating: Float(0), count: frameCount * frameLength)
        var frameMask = Array(repeating: false, count: frameCount)
        for frameIndex in 0 ..< frameCount {
            let start = frameIndex * hopLength
            for sampleIndex in 0 ..< frameLength {
                framed[frameIndex * frameLength + sampleIndex] =
                    waveform[start + sampleIndex] * window[sampleIndex]
            }
            let endIndex = start + frameSizeForUnfold - 1
            frameMask[frameIndex] = attentionMask[endIndex]
        }

        let frames = MLXArray(framed, [frameCount, frameLength])
        let spectrum = rfft(frames, n: fftLength, axis: -1)
        let magnitude = abs(spectrum)
        var logMel = log(matmul(magnitude, melFilters) + melFloor)

        let maskArray = MLXArray(frameMask, [frameCount])
        logMel = logMel * maskArray[0..., .newAxis].asType(logMel.dtype)
        let features = logMel[.newAxis, 0..., 0...]
        let mask = maskArray[.newAxis, 0...]

        return Gemma4PreparedAudio(
            features: features,
            mask: mask,
            durationMilliseconds: durationMilliseconds
        )
    }

    private func resampleIfNeeded(_ samples: [Float], from sourceRate: Int) -> [Float] {
        guard sourceRate != samplingRate, !samples.isEmpty else { return samples }
        let ratio = Double(samplingRate) / Double(sourceRate)
        let outputCount = max(Int(Double(samples.count) * ratio), 1)
        return (0 ..< outputCount).map { index in
            let sourcePosition = Double(index) / ratio
            let lower = min(Int(sourcePosition), samples.count - 1)
            let upper = min(lower + 1, samples.count - 1)
            let fraction = Float(sourcePosition - Double(lower))
            return samples[lower] * (1 - fraction) + samples[upper] * fraction
        }
    }

    private static func makeMelFilterBank(
        frequencyBins: Int,
        melFilters: Int,
        minFrequency: Float,
        maxFrequency: Float,
        samplingRate: Int
    ) -> MLXArray {
        func hzToMel(_ frequency: Float) -> Float {
            2595.0 * log10(1.0 + frequency / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        let melMin = hzToMel(minFrequency)
        let melMax = hzToMel(maxFrequency)
        let melPoints = (0 ..< (melFilters + 2)).map { index in
            melMin + (melMax - melMin) * Float(index) / Float(melFilters + 1)
        }
        let frequencyPoints = melPoints.map(melToHz)
        let allFrequencies = (0 ..< frequencyBins).map { bin in
            Float(bin) * (Float(samplingRate) / (2 * Float(frequencyBins - 1)))
        }

        var filters = Array(repeating: Float(0), count: frequencyBins * melFilters)
        for melIndex in 0 ..< melFilters {
            let lower = frequencyPoints[melIndex]
            let center = frequencyPoints[melIndex + 1]
            let upper = frequencyPoints[melIndex + 2]
            for (frequencyIndex, frequency) in allFrequencies.enumerated() {
                let rising = (frequency - lower) / max(center - lower, 1e-10)
                let falling = (upper - frequency) / max(upper - center, 1e-10)
                filters[frequencyIndex * melFilters + melIndex] = max(0, min(rising, falling))
            }
        }
        return MLXArray(filters, [frequencyBins, melFilters])
    }

    private static func loadMonoSamples(url: URL, sampleRate: Int) throws -> [Float] {
        let source = try AVAudioFile(forReading: url)
        let sourceFormat = source.processingFormat
        guard let sourceBuffer = AVAudioPCMBuffer(
            pcmFormat: sourceFormat,
            frameCapacity: AVAudioFrameCount(source.length)
        ) else {
            return []
        }
        try source.read(into: sourceBuffer)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            return []
        }

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            return []
        }

        var outputSamples: [Float] = []
        outputSamples.reserveCapacity(
            Int(ceil(Double(sourceBuffer.frameLength) * Double(sampleRate) / sourceFormat.sampleRate))
        )
        let inputState = Gemma4AudioConverterInputState()
        var didFinish = false

        while !didFinish {
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: 4096
            ) else {
                return outputSamples
            }

            var conversionError: NSError?
            let conversionStatus = converter.convert(to: outputBuffer, error: &conversionError) {
                _, status in
                if inputState.didProvideInput {
                    status.pointee = .endOfStream
                    return nil
                }
                inputState.didProvideInput = true
                status.pointee = .haveData
                return sourceBuffer
            }
            if let conversionError {
                throw conversionError
            }

            if let channel = outputBuffer.floatChannelData?[0], outputBuffer.frameLength > 0 {
                let count = Int(outputBuffer.frameLength)
                outputSamples.append(
                    contentsOf: UnsafeBufferPointer(start: channel, count: count)
                )
            }

            switch conversionStatus {
            case .haveData, .inputRanDry:
                continue
            case .endOfStream:
                didFinish = true
            case .error:
                didFinish = true
            @unknown default:
                didFinish = true
            }
        }

        return outputSamples
    }
}
