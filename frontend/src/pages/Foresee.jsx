import React, { useEffect, useRef, useState } from "react";

const API_BASE_URL = "http://localhost:5000/api";

function Foresee() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // ✅ STAGES: upload → loading → fadingOut → success
  const [stage, setStage] = useState("upload");

  // ✅ Success header swap + modal
  const [showSuccessHeader, setShowSuccessHeader] = useState(true);
  const [successHeaderFading, setSuccessHeaderFading] = useState(false);
  const [stageAfterSuccess, setStageAfterSuccess] = useState(false); // show target buttons
  const [showModal, setShowModal] = useState(false);

  // Rotating loading messages
  const loadingMessages = [
    "Loading your data...",
    "Preparing for analysis...",
    "Sit back and relax...",
    "Identifying target variable...",
    "Generating insights...",
    "Almost there, your prediction is on the way!",
  ];

  const [activeMsgIdx, setActiveMsgIdx] = useState(0);
  const [prevMsgIdx, setPrevMsgIdx] = useState(null);
  const MESSAGE_FADE_TIME = 250; // 0.25s fade

  // ✅ Fixed non-overlapping fade logic for loading messages
  useEffect(() => {
    if (stage !== "loading") return;

    const interval = setInterval(() => {
      setPrevMsgIdx(activeMsgIdx); // start fade-out of current

      setTimeout(() => {
        setActiveMsgIdx((cur) =>
          cur < loadingMessages.length - 1 ? cur + 1 : cur
        );
        setPrevMsgIdx(null); // remove old message completely
      }, MESSAGE_FADE_TIME);
    }, 2000);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, activeMsgIdx]);

  // ✅ When entering success: wait 2s, fade out header, then show target buttons
  useEffect(() => {
    if (stage === "success") {
      const startFade = setTimeout(() => {
        setSuccessHeaderFading(true); // apply fade-out class
      }, 2000); // show for 2s

      const swapToTargets = setTimeout(() => {
        setShowSuccessHeader(false);  // unmount header
        setStageAfterSuccess(true);   // show target selection
      }, 2000 + 400); // 400ms matches .fade-out duration below

      return () => {
        clearTimeout(startFade);
        clearTimeout(swapToTargets);
      };
    }
  }, [stage]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith(".csv")) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError("Please upload a CSV file");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith(".csv")) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError("Please upload a CSV file");
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setStage("loading");
    setUploading(true);
    setError(null);
    setResult(null);
    setActiveMsgIdx(0);
    setPrevMsgIdx(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("workflow_name", `Analysis - ${file.name}`);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setResult(data);
        setFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";

        // ✅ Fade out the upload section first, then show success
        setStage("fadingOut");
        setTimeout(() => {
          // Reset success header states each time we arrive here
          setShowSuccessHeader(true);
          setSuccessHeaderFading(false);
          setStageAfterSuccess(false);
          setStage("success");
        }, 500); // 0.5s fadeOut you chose
      } else {
        setError(data.error || "Upload failed");
        setStage("upload");
        setFile(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    } catch (err) {
      setError(
        `Connection error: ${err.message}. Make sure the backend is running on port 5000.`
      );
      setStage("upload");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      console.error("Upload error:", err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="font-poppin pt-48">
      {/* ✅ Keyframes and classes */}
      <style>{`
        /* Message fade in */
        @keyframes fadeInMessage {
          0% { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }

        /* Message fade out */
        @keyframes fadeOutMessage {
          0% { opacity: 1; transform: translateY(0); }
          100% { opacity: 0; transform: translateY(-8px); }
        }

        .msg-in { animation: fadeInMessage 0.25s ease forwards; }
        .msg-out { animation: fadeOutMessage 0.25s ease forwards; }

        /* Generic section fades */
        @keyframes fadeInSoft {
          0% { opacity: 0; transform: translateY(12px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeOutSoft {
          0% { opacity: 1; transform: translateY(0); }
          100% { opacity: 0; transform: translateY(-12px); }
        }
        .fade-in { animation: fadeInSoft 0.4s ease forwards; }
        .fade-out { animation: fadeOutSoft 0.4s ease forwards; }
      `}</style>

      {/* ✅ Upload + Loading + FadingOut */}
      {(stage === "upload" || stage === "loading" || stage === "fadingOut") && (
        <section
          className={`w-full bg-[#EEEDE9] py-18 transition-all duration-500 ${
            stage === "fadingOut" ? "opacity-0 -translate-y-4" : "opacity-100 translate-y-0"
          }`}
        >
          <div className="max-w-4xl mx-auto px-8">
            <h2 className="text-3xl font-bold mb-8 text-center text-black">
              Get Started with Your Analysis
            </h2>
            <div className="relative">
              {/* Upload UI */}
              <div
                className={`transition-all duration-300 ${
                  stage === "upload" ? "opacity-100" : "opacity-0 pointer-events-none"
                }`}
              >
                <div
                  className={`border-2 border-dashed rounded-xl p-8 bg-white transition-all ${
                    dragActive ? "border-blue-600 bg-blue-50" : "border-gray-400"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <div className="flex flex-col items-center text-center space-y-6">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={1.5}
                      stroke="currentColor"
                      className={`w-20 h-20 ${
                        dragActive ? "text-blue-600" : "text-gray-600"
                      }`}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
                      />
                    </svg>

                    <div>
                      <p className="text-lg font-semibold text-gray-800 mb-2">
                        Drag & drop your CSV file here
                      </p>
                      <p className="text-sm text-gray-600">or</p>
                    </div>

                    {file && (
                      <div className="w-full max-w-md p-4 bg-gray-100 rounded-lg border border-gray-300">
                        <p className="text-sm font-semibold text-gray-800">
                          Selected File:
                        </p>
                        <p className="text-sm text-gray-700">{file.name}</p>
                        <p className="text-xs text-gray-500 mt-1">
                          Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    )}

                    <div className="flex gap-4">
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        className="hidden"
                        id="file-upload"
                      />

                      <label htmlFor="file-upload">
                        <div className="px-8 py-3 bg-gray-800 text-white rounded-lg cursor-pointer hover:bg-gray-700 transition font-semibold">
                          Choose File
                        </div>
                      </label>

                      {file && (
                        <button
                          onClick={handleUpload}
                          disabled={uploading}
                          className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {uploading ? (
                            <span className="flex items-center gap-2">
                              <svg
                                className="animate-spin h-5 w-5"
                                xmlns="http://www.w3.org/2000/svg"
                                fill="none"
                                viewBox="0 0 24 24"
                              >
                                <circle
                                  className="opacity-25"
                                  cx="12"
                                  cy="12"
                                  r="10"
                                  stroke="currentColor"
                                  strokeWidth="4"
                                ></circle>
                                <path
                                  className="opacity-75"
                                  fill="currentColor"
                                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                ></path>
                              </svg>
                              Uploading...
                            </span>
                          ) : (
                            "Upload & Analyze"
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {error && (
                  <div className="mt-6 p-4 bg-red-100 border-l-4 border-red-500 text-red-700 rounded">
                    <div className="flex items-center">
                      <svg
                        className="w-6 h-6 mr-2"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <div>
                        <p className="font-semibold">Error</p>
                        <p className="text-sm">{error}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* ✅ Loading Overlay */}
              <div
                className={`absolute inset-0 transition-opacity duration-300 flex items-center justify-center ${
                  stage === "loading"
                    ? "opacity-100"
                    : "opacity-0 pointer-events-none"
                }`}
              >
                <div className="w-full">
                  <div className="flex flex-col items-center gap-6 py-16 bg-white rounded-xl border border-gray-200 shadow-sm">
                    <svg
                      className="animate-spin h-10 w-10 text-blue-600"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>

                    {/* ✅ Non-overlapping messages */}
                    <div className="relative flex items-center justify-center h-7 w-full">
                      {prevMsgIdx !== null ? (
                        <p className="msg-out text-gray-500 font-medium text-base text-center px-4">
                          {loadingMessages[prevMsgIdx]}
                        </p>
                      ) : (
                        <p className="msg-in text-gray-800 font-medium text-base text-center px-4">
                          {loadingMessages[activeMsgIdx]}
                        </p>
                      )}
                    </div>

                    <p className="text-xs text-gray-500">
                      Do not close this page while we process your file.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ✅ Success Section */}
      {stage === "success" && result && (
        <section className="w-full bg-[#EEEDE9] py-0">
          <div className="max-w-6xl mx-auto px-8">
            {/* ✅ Success header auto-fades out after 2s, then swaps to target selection */}
            {showSuccessHeader && (
              <div className={`text-center mb-12 ${successHeaderFading ? "fade-out" : "fade-in"}`}>
                <div className="inline-flex items-center justify-center w-16 h-16 bg-green-500 rounded-full mb-4">
                  <svg
                    className="w-10 h-10 text-white"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <h2 className="text-3xl font-bold text-black mb-2">
                  Upload Successful!
                </h2>
                <p className="text-gray-700">
                  Your data has been processed and loaded into Snowflake
                </p>
              </div>
            )}

            {!showSuccessHeader && stageAfterSuccess && (
              <div className="text-center mb-12 fade-in">
                <h2 className="text-2xl font-bold mb-6">
                  Please select a prediction target
                </h2>
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button className="px-6 py-3 bg-blue-600 text-white rounded-lg">
                    Option A
                  </button>
                  <button className="px-6 py-3 bg-blue-600 text-white rounded-lg">
                    Option B
                  </button>
                  <button className="px-6 py-3 bg-blue-600 text-white rounded-lg">
                    Option C
                  </button>
                  <button
                    onClick={() => setShowModal(true)}
                    className="px-6 py-3 bg-gray-700 text-white rounded-lg"
                  >
                    Other Options
                  </button>
                </div>
              </div>
            )}

            {/* ✅ Your existing success details remain unchanged */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
              <div className="bg-white rounded-xl p-6 shadow-lg">
                <div className="text-sm text-gray-600 mb-2">Workflow ID</div>
                <div className="font-mono text-xs text-gray-800 break-all">
                  {result.workflow.id}
                </div>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-lg">
                <div className="text-sm text-gray-600 mb-2">Schema Name</div>
                <div className="font-mono text-xs text-gray-800 break-all">
                  {result.workflow.schema}
                </div>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-lg">
                <div className="text-sm text-gray-600 mb-2">Rows Loaded</div>
                <div className="text-3xl font-bold text-blue-600">
                  {result.upload.rows_loaded.toLocaleString()}
                </div>
              </div>

              <div className="bg-white rounded-xl p-6 shadow-lg">
                <div className="text-sm text-gray-600 mb-2">File Size</div>
                <div className="text-3xl font-bold text-blue-600">
                  {result.upload.file_size_mb} MB
                </div>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold mb-4 text-black">
                Data Preview (First 5 Rows)
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-100">
                    <tr>
                      {result.preview.columns.map((col, idx) => (
                        <th
                          key={idx}
                          className="px-4 py-3 text-left font-semibold text-gray-700 border-b-2 border-gray-300"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.preview.data.map((row, rowIdx) => (
                      <tr
                        key={rowIdx}
                        className="border-b border-gray-200 hover:bg-gray-50 transition"
                      >
                        {row.map((cell, cellIdx) => (
                          <td key={cellIdx} className="px-4 py-3 text-gray-800">
                            {cell !== null ? cell.toString() : "NULL"}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
                <p className="text-sm text-gray-700">
                  <strong>Next Steps:</strong> Your data is now ready for
                  analysis. The workflow has been created and your CSV data is
                  stored in Snowflake table{" "}
                  <code className="bg-gray-200 px-2 py-1 rounded font-mono text-xs">
                    {result.upload.table_name}
                  </code>
                  .
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ✅ Modal for Other Options */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg w-80 text-center shadow-lg">
            <h3 className="text-lg font-semibold mb-4">Select Another Target</h3>
            <ul className="space-y-2 mb-6 text-left">
              <li className="py-2 px-4 bg-gray-100 rounded cursor-pointer">Target 4</li>
              <li className="py-2 px-4 bg-gray-100 rounded cursor-pointer">Target 5</li>
              <li className="py-2 px-4 bg-gray-100 rounded cursor-pointer">Target 6</li>
              <li className="py-2 px-4 bg-gray-100 rounded cursor-pointer">Target 7</li>
            </ul>
            <button
              onClick={() => setShowModal(false)}
              className="px-4 py-2 bg-gray-700 text-white rounded"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Foresee;
