import React, { useEffect, useRef, useState, useCallback } from "react";

const API_BASE_URL = "http://localhost:5000/api";

function Foresee() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // ✅ STAGES: upload → loading → fadingOut → success → modelLoading → modelReady
  const [stage, setStage] = useState("upload");

  // ✅ Success header swap + modal
  const [showSuccessHeader, setShowSuccessHeader] = useState(true);
  const [successHeaderFading, setSuccessHeaderFading] = useState(false);
  const [stageAfterSuccess, setStageAfterSuccess] = useState(false); // show target buttons
  const [showModal, setShowModal] = useState(false);

  // ✅ Target variable suggestions
  const [targetSuggestions, setTargetSuggestions] = useState([]);
  const [loadingTargets, setLoadingTargets] = useState(false);
  const [targetError, setTargetError] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [savingTarget, setSavingTarget] = useState(false);

  // ✅ Fade out entire success section after target selection
  const [fadeAllOut, setFadeAllOut] = useState(false);

  // Rotating loading messages (upload)
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

  // ✅ Model-building loading messages (after target selection)
  const modelMessages = [
    "Training regression models…",
    "Evaluating feature importance…",
    "Optimizing hyperparameters…",
    "Computing accuracy and error metrics…",
    "Finalizing your insights report…"
  ];
  const [modelMsgIdx, setModelMsgIdx] = useState(0);
  const [modelPrevMsgIdx, setModelPrevMsgIdx] = useState(null);

  // timers cleanup
  const timersRef = useRef([]);

  const pushTimer = (id) => {
    timersRef.current.push(id);
  };
  const clearAllTimers = () => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  };

  useEffect(() => () => clearAllTimers(), []);

  // ✅ Define fetchTargetSuggestions with useCallback before it's used
  const fetchTargetSuggestions = useCallback(async () => {
    if (!result?.workflow?.id || !result?.upload?.table_name) {
      console.log("⚠️ Missing workflow or table info, skipping target suggestions");
      return;
    }

    setLoadingTargets(true);
    setTargetError(null);

    try {
      console.log(
        `🎯 Fetching target suggestions for workflow: ${result.workflow.id}, table: ${result.upload.table_name}`
      );
      const response = await fetch(
        `${API_BASE_URL}/target-suggestions/${result.workflow.id}/${result.upload.table_name}`
      );
      const data = await response.json();

      console.log("📡 Target suggestions response:", data);

      if (response.ok) {
        setTargetSuggestions(data.recommendations || []);
        console.log("✅ Target suggestions loaded:", data.recommendations);
      } else {
        const errorMsg = data.error || "Failed to load target suggestions";
        setTargetError(errorMsg);
        console.error("❌ Target suggestions error:", errorMsg);
      }
    } catch (err) {
      const errorMsg = `Failed to fetch target suggestions: ${err.message}`;
      setTargetError(errorMsg);
      console.error("❌ Target fetch error:", err);
    } finally {
      setLoadingTargets(false);
    }
  }, [result]);

  // ✅ Fixed non-overlapping fade logic for loading messages (upload phase)
  useEffect(() => {
    if (stage !== "loading") return;

    const interval = setInterval(() => {
      setPrevMsgIdx(activeMsgIdx); // start fade-out of current

      const t = setTimeout(() => {
        setActiveMsgIdx((cur) => (cur < loadingMessages.length - 1 ? cur + 1 : cur));
        setPrevMsgIdx(null); // remove old message completely
      }, MESSAGE_FADE_TIME);
      pushTimer(t);
    }, 2000);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, activeMsgIdx]);

  // ✅ When entering success: fetch targets, wait 2s, fade out header, then show target buttons
  useEffect(() => {
    if (stage === "success" && result) {
      // Fetch target variable suggestions
      fetchTargetSuggestions();

      const t1 = setTimeout(() => {
        setSuccessHeaderFading(true); // apply fade-out class
      }, 2000); // show for 2s
      pushTimer(t1);

      const t2 = setTimeout(() => {
        setShowSuccessHeader(false); // unmount header
        setStageAfterSuccess(true); // show target selection
      }, 2400); // 400ms matches fade-out duration
      pushTimer(t2);
    }
  }, [stage, result, fetchTargetSuggestions]);

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

  // ✅ After selecting a target: keep banner, then fade everything out after 2s → show model loader → 10s → final screen
  const handleSelectTarget = async (suggestion) => {
    if (!result?.workflow?.id || !result?.upload?.table_name) {
      setError("Missing workflow information");
      return;
    }

    setSavingTarget(true);
    setTargetError(null);

    try {
      console.log(`💾 Saving target selection: ${suggestion.variable}`);

      const response = await fetch(
        `${API_BASE_URL}/workflow/${result.workflow.id}/select-target`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            target_variable: suggestion.variable,
            table_name: result.upload.table_name,
            problem_type: suggestion.problem_type,
            importance_score: suggestion.importance_score,
          }),
        }
      );

      const data = await response.json();

      if (response.ok) {
        setSelectedTarget(suggestion.variable);
        console.log("✅ Target variable saved:", data);

        // ✅ 2s later: fade out all success content
        const t1 = setTimeout(() => {
          setFadeAllOut(true);
        }, 2000);
        pushTimer(t1);

        // After fade (0.4s), switch to modelLoading
        const t2 = setTimeout(() => {
          setStage("modelLoading");
          setFadeAllOut(false); // reset for future runs
          setModelMsgIdx(0);
          setModelPrevMsgIdx(null);
        }, 2000 + 400);
        pushTimer(t2);

        // ✅ Model loader runs for 10 seconds with rotating messages
        const msgInterval = setInterval(() => {
          setModelPrevMsgIdx((prev) => (prev === null ? 0 : modelMsgIdx));
          setModelPrevMsgIdx(modelMsgIdx);
          const t = setTimeout(() => {
            setModelMsgIdx((cur) => (cur < modelMessages.length - 1 ? cur + 1 : 0));
            setModelPrevMsgIdx(null);
          }, MESSAGE_FADE_TIME);
          pushTimer(t);
        }, 2000);

        // stop the interval when we leave modelLoading
        const stopInterval = () => clearInterval(msgInterval);

        // After 10 seconds: fade out loader → show modelReady
        const t3 = setTimeout(() => {
          stopInterval();
          setFadeAllOut(true);
        }, 10000);
        pushTimer(t3);

        const t4 = setTimeout(() => {
          setStage("modelReady");
          setFadeAllOut(false);
        }, 10000 + 400);
        pushTimer(t4);
      } else {
        setTargetError(data.error || "Failed to save target selection");
        console.error("❌ Target selection error:", data.error);
      }
    } catch (err) {
      setTargetError(`Failed to save target: ${err.message}`);
      console.error("❌ Target save error:", err);
    } finally {
      setSavingTarget(false);
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
    setTargetSuggestions([]);
    setTargetError(null);
    setSelectedTarget(null);

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
        const t = setTimeout(() => {
          // Reset success header states each time we arrive here
          setShowSuccessHeader(true);
          setSuccessHeaderFading(false);
          setStageAfterSuccess(false);
          setStage("success");
        }, 500); // 0.5s fadeOut you chose
        pushTimer(t);
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

  // Report handlers (adjust endpoints as needed)
  const handleViewReport = () => {
    if (!result?.workflow?.id) return;
    // open a view endpoint in a new tab
    window.open(`${API_BASE_URL}/workflow/${result.workflow.id}/report/view`, "_blank");
  };

  const handleDownloadReport = () => {
    if (!result?.workflow?.id) return;
    // trigger a file download
    window.location.href = `${API_BASE_URL}/workflow/${result.workflow.id}/report/download`;
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

              {/* ✅ Loading Overlay (upload) */}
              <div
                className={`absolute inset-0 transition-opacity duration-300 flex items-center justify-center ${
                  stage === "loading" ? "opacity-100" : "opacity-0 pointer-events-none"
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
        <section className={`w-full bg-[#EEEDE9] py-0 transition-all duration-400 ${fadeAllOut ? "opacity-0 -translate-y-3" : "opacity-100 translate-y-0"}`}>
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
                
                {loadingTargets && (
                  <div className="flex items-center justify-center gap-2 mb-4">
                    <svg
                      className="animate-spin h-5 w-5 text-blue-600"
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
                    <span className="text-gray-600">Loading target recommendations...</span>
                  </div>
                )}

                {targetError && (
                  <div className="mb-4 p-3 bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 text-sm rounded">
                    {targetError}
                  </div>
                )}

                {selectedTarget && (
                  <div className="mb-4 p-4 bg-green-100 border-l-4 border-green-500 text-green-700 rounded">
                    <div className="flex items-center">
                      <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                      </svg>
                      <div>
                        <p className="font-semibold">Target Selected!</p>
                        <p className="text-sm">Target variable "{selectedTarget}" saved for model training</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* ✅ Podium-style target buttons */}
                <div className="flex flex-col items-center mt-6 gap-4">
                  <div className="flex items-end justify-center gap-4">
                    {/* Silver */}
                    {targetSuggestions.length >= 2 && (
                      <button
                        onClick={() => handleSelectTarget(targetSuggestions[1])}
                        disabled={savingTarget || selectedTarget === targetSuggestions[1].variable}
                        className={`px-8 py-4 rounded-lg transition transform ${
                          selectedTarget === targetSuggestions[1].variable
                            ? 'bg-green-600 text-white'
                            : 'bg-gray-300 text-black hover:bg-gray-400'
                        } disabled:opacity-50 disabled:cursor-not-allowed scale-90`}
                      >
                        <div className="flex flex-col items-center">
                          <span className="font-semibold">{targetSuggestions[1].variable}</span>
                          <span className="text-xs opacity-80">{targetSuggestions[1].problem_type}</span>
                        </div>
                      </button>
                    )}
                    {/* Gold */}
                    {targetSuggestions.length >= 1 && (
                      <button
                        onClick={() => handleSelectTarget(targetSuggestions[0])}
                        disabled={savingTarget || selectedTarget === targetSuggestions[0].variable}
                        className={`px-12 py-6 rounded-lg transition transform ${
                          selectedTarget === targetSuggestions[0].variable
                            ? 'bg-green-600 text-white'
                            : 'bg-yellow-500 text-black hover:bg-yellow-600'
                        } disabled:opacity-50 disabled:cursor-not-allowed scale-110`}
                      >
                        <div className="flex flex-col items-center">
                          <span className="font-bold">{targetSuggestions[0].variable}</span>
                          <span className="text-xs opacity-80">{targetSuggestions[0].problem_type}</span>
                        </div>
                      </button>
                    )}
                    {/* Bronze */}
                    {targetSuggestions.length >= 3 && (
                      <button
                        onClick={() => handleSelectTarget(targetSuggestions[2])}
                        disabled={savingTarget || selectedTarget === targetSuggestions[2].variable}
                        className={`px-8 py-4 rounded-lg transition transform ${
                          selectedTarget === targetSuggestions[2].variable
                            ? 'bg-green-600 text-white'
                            : 'bg-amber-700 text-white hover:bg-amber-800'
                        } disabled:opacity-50 disabled:cursor-not-allowed scale-90`}
                      >
                        <div className="flex flex-col items-center">
                          <span className="font-semibold">{targetSuggestions[2].variable}</span>
                          <span className="text-xs opacity-80">{targetSuggestions[2].problem_type}</span>
                        </div>
                      </button>
                    )}
                  </div>

                  {/* Other options */}
                  {targetSuggestions.length > 3 && (
                    <button
                      onClick={() => {
                        if (!savingTarget && !selectedTarget) {
                          setShowModal(true);
                        }
                      }}
                      disabled={savingTarget || selectedTarget !== null}
                     className={`px-6 py-3 rounded-lg transition font-semibold ${
                      savingTarget || selectedTarget !== null
                        ? "bg-gray-400 text-white opacity-50 cursor-not-allowed"
                        : "bg-gray-700 text-white hover:bg-gray-600"
                    }`}

                    >
                      Other Options ({targetSuggestions.length - 3} more)
                    </button>
                  )}

                  {/* If no recommendations yet */}
                  {targetSuggestions.length === 0 && !loadingTargets && (
                    <button
                      onClick={() => setShowModal(true)}
                      className="px-6 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
                    >
                      Select Target Variable
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* ✅ Success details */}
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

      {/* ✅ Model Loading Screen */}
      {stage === "modelLoading" && (
        <section className={`w-full bg-[#EEEDE9] py-24 transition-all duration-400 ${fadeAllOut ? "opacity-0 -translate-y-3" : "opacity-100 translate-y-0"}`}>
          <div className="max-w-3xl mx-auto px-8">
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-10 text-center fade-in">
              <svg
                className="animate-spin h-10 w-10 text-blue-600 mx-auto mb-6"
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

              <h2 className="text-2xl font-bold mb-2 text-black">
                Building your regression models…
              </h2>
              <p className="text-gray-600 mb-8">
                We’re training, evaluating, and preparing your report.
              </p>

              {/* Rotating model messages */}
              <div className="relative h-6">
                {modelPrevMsgIdx !== null ? (
                  <p className="msg-out text-gray-500 font-medium text-sm text-center px-4">
                    {modelMessages[modelPrevMsgIdx]}
                  </p>
                ) : (
                  <p className="msg-in text-gray-800 font-medium text-sm text-center px-4">
                    {modelMessages[modelMsgIdx]}
                  </p>
                )}
              </div>

              {/* Simple progress bar illusion */}
              <div className="mt-8 w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                <div className="h-2 bg-blue-600 animate-pulse" style={{ width: "70%" }} />
              </div>

              <p className="text-xs text-gray-500 mt-6">
                This usually takes a moment. Thanks for your patience.
              </p>
            </div>
          </div>
        </section>
      )}

      {/* ✅ Model Ready / Confirmation */}
      {stage === "modelReady" && (
        <section className="w-full bg-[#EEEDE9] py-24">
          <div className="max-w-3xl mx-auto px-8">
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-10 text-center fade-in">
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

              <h2 className="text-3xl font-bold mb-4 text-black">Analysis complete</h2>
              <p className="text-gray-700 mb-8">
                Your regression models have been created and your insights report is ready.
              </p>

              <div className="flex justify-center gap-6">
                <button
                  onClick={handleViewReport}
                  className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
                >
                  View Report
                </button>
                <button
                  onClick={handleDownloadReport}
                  className="px-8 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition"
                >
                  Download Report
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ✅ Modal for Other Options */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white p-6 rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto shadow-xl">
            <h3 className="text-xl font-bold mb-4 text-center">
              All Target Variable Recommendations
            </h3>
            <p className="text-sm text-gray-600 mb-6 text-center">
              Ranked by importance for machine learning prediction
            </p>
            
            {targetSuggestions.length > 0 ? (
              <ul className="space-y-3 mb-6">
                {targetSuggestions.map((suggestion, idx) => (
                  <li
                    key={idx}
                    onClick={() => {
                      setShowModal(false);
                      handleSelectTarget(suggestion);
                    }}
                    className={`p-4 rounded-lg border cursor-pointer transition ${
                      selectedTarget === suggestion.variable
                        ? 'bg-green-50 border-green-300'
                        : 'bg-gray-50 border-gray-200 hover:bg-blue-50 hover:border-blue-300'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-bold text-blue-600">#{suggestion.rank}</span>
                          <span className="font-semibold text-gray-800">
                            {suggestion.variable}
                          </span>
                          {suggestion.importance_score && (
                            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                              Score: {suggestion.importance_score}/100
                            </span>
                          )}
                          {selectedTarget === suggestion.variable && (
                            <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded font-medium">
                              ✓ Selected
                            </span>
                          )}
                        </div>
                        <div className="text-sm text-gray-600 mb-2">
                          <span className="font-medium">Type:</span> {suggestion.problem_type}
                        </div>
                        {suggestion.why_important && (
                          <div className="text-sm text-gray-700 mb-2">
                            <span className="font-medium">Why Important:</span> {suggestion.why_important}
                          </div>
                        )}
                        {suggestion.predictability && (
                          <div className="text-xs text-gray-500">
                            <span className="font-medium">Predictability:</span> {suggestion.predictability}
                          </div>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-center py-8 text-gray-500">
                No target suggestions available yet
              </div>
            )}
            
            <button
              onClick={() => setShowModal(false)}
              className="w-full px-4 py-3 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition font-semibold"
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
