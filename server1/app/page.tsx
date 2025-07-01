"use client";
import { useState } from "react";

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<string>("");
  const [downloadUrl, setDownloadUrl] = useState<string>("");

  const handleUpload = async () => {
    if (!file) return;

    setIsProcessing(true);
    setResult("");
    setDownloadUrl("");

    const formData = new FormData();
    formData.append("audio", file);

    try {
      // Make request directly to FastAPI encode server
      const response = await fetch("http://localhost:8000/encode", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Send encoded data to second server
      const decodeResponse = await fetch("http://localhost:8001/decode", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          encoded_data: data.encoded_data,
          sample_rate: data.sample_rate,
          channels: data.channels,
        }),
      });

      if (!decodeResponse.ok) {
        throw new Error(`Decode error! status: ${decodeResponse.status}`);
      }

      const decodeResult = await decodeResponse.json();
      setResult(decodeResult.message);

      if (decodeResult.download_url) {
        setDownloadUrl(decodeResult.download_url);
      }
    } catch (error) {
      console.error("Error:", error);
      setResult(
        `Error: ${error instanceof Error ? error.message : "Unknown error"}`
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

      <div className="mb-4">
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="mb-4 block"
        />

        <button
          onClick={handleUpload}
          disabled={!file || isProcessing}
          className="bg-blue-500 text-white px-4 py-2 rounded disabled:opacity-50 mr-4"
        >
          {isProcessing ? "Processing..." : "Upload and Process"}
        </button>
      </div>

      {result && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <p>{result}</p>

          {downloadUrl && (
            <button
              onClick={handleDownload}
              className="mt-2 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
            >
              Download Decoded Audio
            </button>
          )}
        </div>
      )}
    </div>
  );
}
