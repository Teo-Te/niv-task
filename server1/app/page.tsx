"use client";
import { useState, useEffect } from "react";
import * as ort from "onnxruntime-web";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<string>("");
  const [downloadUrl, setDownloadUrl] = useState<string>("");
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [ffmpeg, setFFmpeg] = useState<FFmpeg | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const modelResponse = await fetch(
          "http://localhost:8000/get-onnx-model"
        );
        if (!modelResponse.ok) {
          await fetch("http://localhost:8000/convert-to-onnx", {
            method: "POST",
          });
        }

        const session = await ort.InferenceSession.create(
          "http://localhost:8000/model/encodec_encoder.onnx"
        );
        setSession(session);
      } catch (error) {
        setResult("Model loading failed");
      }
    };

    loadModel();
  }, []);

  const initializeFFmpeg = async () => {
    if (ffmpeg) return ffmpeg;

    const ffmpegInstance = new FFmpeg();
    const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd";

    await ffmpegInstance.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
      wasmURL: await toBlobURL(
        `${baseURL}/ffmpeg-core.wasm`,
        "application/wasm"
      ),
    });

    setFFmpeg(ffmpegInstance);
    return ffmpegInstance;
  };

  const convertAudioTo24kHz = async (
    audioFile: File
  ): Promise<Float32Array> => {
    const ffmpegInstance = await initializeFFmpeg();

    await ffmpegInstance.writeFile("input.wav", await fetchFile(audioFile));
    await ffmpegInstance.exec([
      "-i",
      "input.wav",
      "-ar",
      "24000",
      "-ac",
      "1",
      "-f",
      "f32le",
      "output.raw",
    ]);

    const data = (await ffmpegInstance.readFile("output.raw")) as any;
    return new Float32Array(data.buffer);
  };

  const handleUpload = async () => {
    if (!file || !session) return;

    setIsProcessing(true);
    setResult("Processing...");
    setDownloadUrl("");

    try {
      const audioData = await convertAudioTo24kHz(file);
      const chunkSize = 45000;
      const numChunks = Math.ceil(audioData.length / chunkSize);
      const allEncodedChunks = [];

      for (let i = 0; i < numChunks; i++) {
        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, audioData.length);
        let chunkData = audioData.slice(start, end);

        if (chunkData.length < chunkSize) {
          const paddedChunk = new Float32Array(chunkSize);
          paddedChunk.set(chunkData);
          chunkData = paddedChunk;
        }

        const inputTensor = new ort.Tensor("float32", chunkData, [
          1,
          1,
          chunkSize,
        ]);
        const results = await session.run({ audio: inputTensor });

        allEncodedChunks.push({
          chunk_index: i,
          codes: Array.from(results.codes.data as ArrayLike<number>),
          scale: Array.from(results.scale.data as ArrayLike<number>)[0],
          structure: {
            n_q: Array.from(results.n_q.data as ArrayLike<number>)[0],
            channels: Array.from(results.channels.data as ArrayLike<number>)[0],
            time_steps: Array.from(
              results.time_steps.data as ArrayLike<number>
            )[0],
          },
          original_length: end - start,
          was_padded: chunkData.length > end - start,
        });
      }

      const decodeResponse = await fetch("http://localhost:8001/decode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          encoded_data: [
            {
              encoding_method: "onnx_runtime_web_chunks_client",
              chunks: allEncodedChunks,
              num_chunks: numChunks,
              client_encoded: true,
            },
          ],
          sample_rate: 24000,
          channels: 1,
        }),
      });

      if (!decodeResponse.ok) {
        throw new Error("Decode failed");
      }

      const decodeResult = await decodeResponse.json();
      setResult("Done");

      if (decodeResult.download_url) {
        setDownloadUrl(decodeResult.download_url);
      }
    } catch (error) {
      setResult("Error");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (downloadUrl) {
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = "audio.wav";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="p-8 max-w-md mx-auto">
      <h1 className="text-xl mb-6">Audio Processor</h1>

      <input
        type="file"
        accept="audio/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="mb-4 block w-full text-sm file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-gray-100"
      />

      <button
        onClick={handleUpload}
        disabled={!file || !session || isProcessing}
        className="w-full bg-black text-white py-2 rounded disabled:opacity-50 mb-4"
      >
        {isProcessing ? "Processing..." : "Process"}
      </button>

      {result && <p className="text-sm text-gray-600 mb-2">{result}</p>}

      {downloadUrl && (
        <button
          onClick={handleDownload}
          className="w-full bg-green-600 text-white py-2 rounded"
        >
          Download
        </button>
      )}
    </div>
  );
}
