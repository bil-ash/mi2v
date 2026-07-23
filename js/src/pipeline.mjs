/** Runtime-neutral MobileI2V ONNX pipeline.  The model I/O names are part of
 * the export contract in tools/export_mobilei2v_onnx.py. */
export const MODEL_FILES = Object.freeze({
  text: 'qwen2_0.5b.int8.qdq.onnx', denoiser: 'denoiser.int8.qdq.onnx',
  imageEncoder: 'ltx_vae_encoder.int8.qdq.onnx', decoder: 'turbo_vaed_decoder.int8.qdq.onnx',
});

export function makeNoise(length, seed = 1) {
  // Deterministic Mulberry32: identical input noise in Node and a browser.
  let state = seed >>> 0; const next = () => ((state = (state + 0x6D2B79F5) >>> 0), state);
  const uniform = () => { let t = next(); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 2) { const r = Math.sqrt(-2 * Math.log(Math.max(uniform(), 1e-7))); const a = 2 * Math.PI * uniform(); out[i] = r * Math.cos(a); if (i + 1 < length) out[i + 1] = r * Math.sin(a); }
  return out;
}

export function eulerFlow(latents, prediction, dt) {
  const result = new Float32Array(latents.length);
  for (let i = 0; i < result.length; ++i) result[i] = latents[i] + dt * prediction[i];
  return result;
}

export async function runPipeline({ ort, sessions, inputIds, attentionMask, image, shape, steps = 4, seed = 1, flowScore = 2 }) {
  const tensor = (data, dims, type = 'float32') => new ort.Tensor(type, data, dims);
  const text = await sessions.text.run({ input_ids: tensor(inputIds, [1, inputIds.length], 'int64'), attention_mask: tensor(attentionMask, [1, attentionMask.length], 'int64') });
  const imageLatents = (await sessions.imageEncoder.run({ sample: image })).latents;
  const latentSize = shape.reduce((a, b) => a * b, 1);
  let latents = makeNoise(latentSize, seed);
  latents.set(imageLatents.data.subarray(0, Math.min(imageLatents.data.length, latents.length)));
  const nullText = new Float32Array(text.last_hidden_state.data.length);
  for (let i = 0; i < steps; i++) {
    const t = new Float32Array([1 - i / steps]);
    const common = { latent: tensor(latents, shape), timestep: tensor(t, [1]), encoder_hidden_states: text.last_hidden_state, flow_score: tensor(new Float32Array([flowScore]), [1]) };
    const conditional = (await sessions.denoiser.run(common)).velocity.data;
    const unconditional = (await sessions.denoiser.run({ ...common, encoder_hidden_states: tensor(nullText, text.last_hidden_state.dims) })).velocity.data;
    const guided = new Float32Array(conditional.length);
    for (let j = 0; j < guided.length; j++) guided[j] = unconditional[j] + 4.5 * (conditional[j] - unconditional[j]);
    latents = eulerFlow(latents, guided, -1 / steps);
  }
  return (await sessions.decoder.run({ latent: tensor(latents, shape) })).video;
}
