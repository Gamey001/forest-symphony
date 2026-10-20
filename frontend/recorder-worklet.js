// Runs on the browser's audio thread and forwards raw microphone samples to the page.
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) this.port.postMessage(channel.slice());
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);
