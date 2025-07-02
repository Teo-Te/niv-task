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
  const [modelStatus, setModelStatus] = useState<string>("Loading model...");
  const [ffmpeg, setFFmpeg] = useState<FFmpeg | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setModelStatus("Loading ONNX model...");

        const modelResponse = await fetch(
          "http://localhost:8000/get-onnx-model"
        );
        if (!modelResponse.ok) {
          setModelStatus("Converting PyTorch model to ONNX...");
          const convertResponse = await fetch(
            "http://localhost:8000/convert-to-onnx",
            {
              method: "POST",
            }
          );
          if (!convertResponse.ok) {
            throw new Error("Failed to convert model to ONNX");
          }
        }

        const session = await ort.InferenceSession.create(
          "http://localhost:8000/model/encodec_encoder.onnx"
        );
        setSession(session);
        setModelStatus("ONNX model loaded successfully ✓");
      } catch (error) {
        setModelStatus(`Failed to load model: ${error}`);
      }
    };

    loadModel();
  }, []);

  const initializeFFmpeg = async () => {
    if (ffmpeg) return ffmpeg;

    setResult("Initializing FFmpeg...");
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
    setResult("FFmpeg initialized ✓");
    return ffmpegInstance;
  };

  const convertAudioTo24kHz = async (
    audioFile: File
  ): Promise<Float32Array> => {
    const ffmpegInstance = await initializeFFmpeg();
    setResult("Converting audio to 24kHz mono with FFmpeg...");

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
    const audioBuffer = new Float32Array(data.buffer);

    setResult(`Audio converted: ${audioBuffer.length} samples at 24kHz ✓`);
    return audioBuffer;
  };

  const handleUpload = async () => {
    if (!file || !session) {
      setResult("Please select a file and ensure model is loaded");
      return;
    }

    setIsProcessing(true);
    setResult("");
    setDownloadUrl("");

    try {
      const audioData = await convertAudioTo24kHz(file);

      const chunkSize = 45000;
      const totalSamples = audioData.length;
      const numChunks = Math.ceil(totalSamples / chunkSize);

      setResult(`Processing ${numChunks} chunks with ONNX Runtime...`);

      const allEncodedChunks = [];

      for (let i = 0; i < numChunks; i++) {
        setResult(`Encoding chunk ${i + 1}/${numChunks} with ONNX Runtime...`);

        const start = i * chunkSize;
        const end = Math.min(start + chunkSize, totalSamples);
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

        const results = await session.run({
          audio: inputTensor,
        });

        const codesArray = Array.from(results.codes.data as ArrayLike<number>);
        const scaleArray = Array.from(results.scale.data as ArrayLike<number>);
        const nqArray = Array.from(results.n_q.data as ArrayLike<number>);
        const channelsArray = Array.from(
          results.channels.data as ArrayLike<number>
        );
        const timeStepsArray = Array.from(
          results.time_steps.data as ArrayLike<number>
        );

        allEncodedChunks.push({
          chunk_index: i,
          codes: codesArray,
          scale: scaleArray[0],
          structure: {
            n_q: nqArray[0],
            channels: channelsArray[0],
            time_steps: timeStepsArray[0],
          },
          original_length: end - start,
          was_padded: chunkData.length > end - start,
        });
      }

      setResult("Sending client-encoded data to decode server...");

      const encodedData = [
        {
          encoding_method: "onnx_runtime_web_chunks_client",
          chunks: allEncodedChunks,
          num_chunks: numChunks,
          client_encoded: true,
          total_samples: totalSamples,
          chunk_size: chunkSize,
        },
      ];

      const decodeResponse = await fetch("http://localhost:8001/decode", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          encoded_data: encodedData,
          sample_rate: 24000,
          channels: 1,
        }),
      });

      if (!decodeResponse.ok) {
        const errorText = await decodeResponse.text();
        throw new Error(
          `Decode error! status: ${decodeResponse.status}, message: ${errorText}`
        );
      }

      const decodeResult = await decodeResponse.json();
      setResult(`✅ ${decodeResult.message}`);

      if (decodeResult.download_url) {
        setDownloadUrl(decodeResult.download_url);
      }
    } catch (error) {
      setResult(
        `❌ Error: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (downloadUrl) {
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = "decoded_audio.wav";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="p-8">
      <h1 className="text-2xl mb-4">Audio Encoder/Decoder</h1>

      <div className="mb-4 p-4 bg-blue-50 rounded">
        <h2 className="text-lg font-semibold mb-2">Model Status</h2>
        <p className={`${session ? "text-green-600" : "text-orange-600"}`}>
          {modelStatus}
        </p>
      </div>

      <div className="mb-4">
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
        />

        <button
          onClick={handleUpload}
          disabled={!file || !session || isProcessing}
          className="bg-blue-500 text-white px-6 py-2 rounded disabled:opacity-50 disabled:cursor-not-allowed mr-4 hover:bg-blue-600"
        >
          {isProcessing ? "Processing..." : "Process Audio"}
        </button>

        {!session && (
          <span className="text-orange-600 text-sm">Model not ready</span>
        )}
      </div>

      {result && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <p className="mb-3">{result}</p>

          {downloadUrl && (
            <button
              onClick={handleDownload}
              className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
            >
              Download Audio
            </button>
          )}
        </div>
      )}
    </div>
  );
}
