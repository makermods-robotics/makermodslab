// Wire-format probe: prints each external camera's ACTIVE AVFoundation format
// (FourCC + resolution + fps range) so we can tell whether a slow stream landed
// on `yuvs` (raw YUY2 on the wire, fps hard-capped by USB2 bandwidth) or
// `420v` (MJPEG on the wire, decoded by the driver — 30fps capable).
//
// Usage (run from a camera-authorized terminal):
//   swiftc -O debug/format_probe.swift -o debug/format_probe   # once
//   ./debug/format_probe            # active format per camera
//   ./debug/format_probe --formats  # also dump every supported format variant
//
// The interesting read is WHILE the slow (~2fps) stream is live in another
// process: `yuvs` at the streaming resolution → hypothesis 2 confirmed;
// `420v` → wire format is fine, look at exposure (debug/exposure_probe.py).

import AVFoundation
import CoreMedia

func fourcc(_ code: FourCharCode) -> String {
    let bytes: [UInt8] = [
        UInt8((code >> 24) & 0xFF), UInt8((code >> 16) & 0xFF),
        UInt8((code >> 8) & 0xFF), UInt8(code & 0xFF),
    ]
    return String(bytes: bytes, encoding: .ascii) ?? String(format: "0x%08x", code)
}

func describe(_ format: AVCaptureDevice.Format) -> String {
    let desc = format.formatDescription
    let sub = fourcc(CMFormatDescriptionGetMediaSubType(desc))
    let dims = CMVideoFormatDescriptionGetDimensions(desc)
    let ranges = format.videoSupportedFrameRateRanges
        .map { String(format: "%.1f-%.1f", $0.minFrameRate, $0.maxFrameRate) }
        .joined(separator: ",")
    return "\(sub) \(dims.width)x\(dims.height) fps=\(ranges)"
}

let dumpAll = CommandLine.arguments.contains("--formats")

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.external], mediaType: .video, position: .unspecified)

let devices = discovery.devices
if devices.isEmpty {
    print("no external cameras visible (TCC block, or none attached)")
}

for device in devices {
    print("\(device.localizedName)  uniqueID=\(device.uniqueID)")
    print("  ACTIVE: \(describe(device.activeFormat))")
    let minDur = device.activeVideoMinFrameDuration
    let maxDur = device.activeVideoMaxFrameDuration
    if minDur.isNumeric, minDur.seconds > 0 {
        print(String(format: "  frame-duration clamp: %.1f fps max", 1.0 / minDur.seconds))
    }
    if maxDur.isNumeric, maxDur.seconds > 0 {
        print(String(format: "  frame-duration clamp: %.1f fps min", 1.0 / maxDur.seconds))
    }
    if dumpAll {
        for format in device.formats {
            print("    \(describe(format))")
        }
    }
}
