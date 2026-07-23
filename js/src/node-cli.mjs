import fs from 'node:fs/promises';
import path from 'node:path';
import { createCanvas, loadImage } from '@napi-rs/canvas';
import ort from 'onnxruntime-node';
import ffmpeg from '@mmomtchev/ffmpeg';
import { VideoEncoder, Muxer } from '@mmomtchev/ffmpeg/stream';
import { MODEL_FILES, runPipeline } from './pipeline.mjs';

function option(name, fallback) { const at = process.argv.indexOf(`--${name}`); return at < 0 ? fallback : process.argv[at + 1]; }
function usage() { console.error('Usage: npm run node -- --image input.png --prompt "..." --models models --output results/video.mp4 [--seed 1]'); process.exit(2); }
async function imageTensor(file, width = 1280, height = 720) { const canvas = createCanvas(width, height); const ctx = canvas.getContext('2d'); ctx.drawImage(await loadImage(file), 0, 0, width, height); const rgba = ctx.getImageData(0, 0, width, height).data; const data = new Float32Array(3 * height * width); for (let y=0;y<height;y++) for(let x=0;x<width;x++) for(let c=0;c<3;c++) data[c*width*height+y*width+x]=rgba[(y*width+x)*4+c]/127.5-1; return new ort.Tensor('float32', data, [1,3,1,height,width]); }
function tokenize(prompt, size = 300) { const ids = new BigInt64Array(size); const mask = new BigInt64Array(size); for (let i = 0; i < Math.min(prompt.length, size); ++i) { ids[i] = BigInt(prompt.codePointAt(i)); mask[i] = 1n; } return { ids, mask }; }
async function encodeMp4(video, output, width = 1280, height = 720, fps = 17) { const pixelFormat = new ffmpeg.PixelFormat('rgb24'); const timeBase = new ffmpeg.Rational(1, fps); const encoder = new VideoEncoder({type:'Video', codec:ffmpeg.AV_CODEC_H264, bitRate:5e6, width, height, frameRate:new ffmpeg.Rational(fps,1), timeBase, pixelFormat}); const muxer = new Muxer({outputFile:output, outputFormat:'mp4', streams:[encoder]}); encoder.pipe(muxer.video[0]); await new Promise((resolve,reject) => { muxer.once('finish', resolve); encoder.once('error',reject); muxer.once('error',reject); for(let t=0;t<video.dims[2];t++) { const rgb=Buffer.alloc(width*height*3); for(let y=0;y<height;y++) for(let x=0;x<width;x++) for(let c=0;c<3;c++) rgb[(y*width+x)*3+c]=Math.max(0,Math.min(255,Math.round((video.data[((c*video.dims[2]+t)*height+y)*width+x]+1)*127.5))); const frame=ffmpeg.VideoFrame.create(rgb,pixelFormat,width,height); frame.setTimeBase(timeBase); frame.setPts(new ffmpeg.Timestamp(t,timeBase)); encoder.write(frame,'binary'); } encoder.end(); }); }
const imageFile=option('image'), prompt=option('prompt'), modelDir=option('models','models'), output=option('output','results/mobilei2v.mp4'); if(!imageFile||!prompt) usage();
await fs.mkdir(path.dirname(output), {recursive:true});
const sessions=Object.fromEntries(await Promise.all(Object.entries(MODEL_FILES).map(async ([key,file]) => [key, await ort.InferenceSession.create(path.join(modelDir,file), {executionProviders:['cpu']} )])));
const tokens=tokenize(prompt); const video=await runPipeline({ort,sessions,inputIds:tokens.ids,attentionMask:tokens.mask,image:await imageTensor(imageFile),shape:[1,128,3,90,160],seed:Number(option('seed','1'))}); await encodeMp4(video,output); console.log(`Saved ${output}`);
