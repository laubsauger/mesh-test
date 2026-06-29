// Webcam capture → a <video> element ready for frame grabbing. Display mirroring
// is a CSS concern for the caller; decode runs on un-mirrored pixels (drawImage
// ignores CSS transforms), matching the reference's model space (V6).
export async function startWebcam({ width = 1280, height = 720 } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('getUserMedia unavailable — webcam pose needs a secure context (https or localhost)');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: width }, height: { ideal: height }, facingMode: 'user' },
    audio: false
  });
  const video = document.createElement('video');
  video.playsInline = true;
  video.muted = true;
  video.srcObject = stream;
  await video.play();
  return { video, stream };
}

export function stopWebcam(stream) {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
}
