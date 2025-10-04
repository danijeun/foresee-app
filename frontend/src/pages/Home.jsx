import React, { useState, useRef } from "react";
import homeimage1 from "../assets/homeimage1.png";

const API_BASE_URL = 'http://localhost:5000/api';

function Home() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError('Please upload a CSV file');
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('Please upload a CSV file');
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('workflow_name', `Analysis - ${file.name}`);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setResult(data);
        setFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } else {
        setError(data.error || 'Upload failed');
      }
    } catch (err) {
      setError(`Connection error: ${err.message}. Make sure the backend is running on port 5000.`);
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="font-poppin pt-32">
      
      {/* ✅ Hero Section (full width) */}
      <section className="w-full bg-[#EEEDE9]" data-aos="fade-up">
        <div className="flex flex-col md:flex-row justify-between items-center max-w-6xl mx-auto px-8 py-20 gap-12">
          <div className="max-w-xl">
            <h2 className="text-2xl md:text-3xl font-bold mb-6 text-black">
              Smart analysis, simplified
            </h2>
            <p className="text-gray-700 leading-relaxed">
              Upload your CSV file and let our AI-powered platform analyze your data, train multiple ML models, and generate comprehensive insights instantly.
            </p>
          </div>

          <div className="mt-10 md:mt-0 w-full md:w-2/5">
            <img
              src={homeimage1}
              alt="Data Visual"
              className="w-full rounded-md shadow-md"
            />
          </div>
        </div>
      </section>

      {/* ✅ How it works Section */}
      <section className="w-full bg-[#D9D7D3] py-16 text-center" data-aos="fade-up">
        <h2 className="text-3xl font-bold mb-4 text-black">
          Data shouldn’t be too hard to understand!
        </h2>
        <h3 className="text-xl font-semibold mb-12">How it works</h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 px-12 max-w-6xl mx-auto">

          {/* Step 1 */}
          <div
            className="flex flex-col items-center text-center space-y-4 bg-[#E6E4DF] shadow-md rounded-xl p-6 animate-zoom"
            style={{ animationDelay: "0s" }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
            <h4 className="text-lg font-bold">1. Upload</h4>
            <p className="text-sm text-gray-800">Upload the CSV file you want to analyze</p>
          </div>

          {/* Step 2 */}
          <div
            className="flex flex-col items-center text-center space-y-4 bg-[#E6E4DF] shadow-md rounded-xl p-6 animate-zoom"
            style={{ animationDelay: "2.5s" }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 4.5v15m6-15v15m-10.875 0h15.75c.621 0 1.125-.504 1.125-1.125V5.625c0-.621-.504-1.125-1.125-1.125H4.125C3.504 4.5 3 5.004 3 5.625v12.75c0 .621.504 1.125 1.125 1.125Z" />
            </svg>
            <h4 className="text-lg font-bold">2. Choose</h4>
            <p className="text-sm text-gray-800">Choose what you want to predict</p>
          </div>

          {/* Step 3 */}
          <div
            className="flex flex-col items-center text-center space-y-4 bg-[#E6E4DF] shadow-md rounded-xl p-6 animate-zoom"
            style={{ animationDelay: "5s" }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 0 0-2.456 2.456ZM16.894 20.567 16.5 21.75l-.394-1.183a2.25 2.25 0 0 0-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 0 0 1.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 0 0 1.423 1.423l1.183.394-1.183.394a2.25 2.25 0 0 0-1.423 1.423Z" />
            </svg>
            <h4 className="text-lg font-bold">3. Relax</h4>
            <p className="text-sm text-gray-800">Sit back while we work on your analysis</p>
          </div>

          {/* Step 4 */}
          <div
            className="flex flex-col items-center text-center space-y-4 bg-[#E6E4DF] shadow-md rounded-xl p-6 animate-zoom"
            style={{ animationDelay: "7.5s" }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Zm3.75 11.625a2.625 2.625 0 1 1-5.25 0 2.625 2.625 0 0 1 5.25 0Z" />
            </svg>
            <h4 className="text-lg font-bold">4. Analyze</h4>
            <p className="text-sm text-gray-800">Your analysis is ready!</p>
          </div>

        </div>
      </section>

      {/* ✅ Upload Section */}
      <section className="w-full bg-[#EEEDE9] py-20" data-aos="fade-up">
        <div className="max-w-4xl mx-auto px-8">
          <h2 className="text-3xl font-bold mb-8 text-center text-black">
            Get Started with Your Analysis
          </h2>

          {/* Upload Box */}
          <div 
            className={`border-2 border-dashed rounded-xl p-12 bg-white transition-all ${
              dragActive ? 'border-blue-600 bg-blue-50' : 'border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center text-center space-y-6">
              {/* Upload Icon */}
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                strokeWidth={1.5} 
                stroke="currentColor" 
                className={`w-20 h-20 ${dragActive ? 'text-blue-600' : 'text-gray-600'}`}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
              </svg>

              <div>
                <p className="text-lg font-semibold text-gray-800 mb-2">
                  Drag & drop your CSV file here
                </p>
                <p className="text-sm text-gray-600">or</p>
              </div>

              {/* File Info */}
              {file && (
                <div className="w-full max-w-md p-4 bg-gray-100 rounded-lg border border-gray-300">
                  <p className="text-sm font-semibold text-gray-800">Selected File:</p>
                  <p className="text-sm text-gray-700">{file.name}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    Size: {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              )}

              {/* Buttons */}
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
                        <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Uploading...
                      </span>
                    ) : (
                      'Upload & Analyze'
                    )}
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-6 p-4 bg-red-100 border-l-4 border-red-500 text-red-700 rounded">
              <div className="flex items-center">
                <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd"/>
                </svg>
                <div>
                  <p className="font-semibold">Error</p>
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* ✅ Results Section */}
      {result && (
        <section className="w-full bg-[#D9D7D3] py-20" data-aos="fade-up">
          <div className="max-w-6xl mx-auto px-8">
            {/* Success Header */}
            <div className="text-center mb-12">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-green-500 rounded-full mb-4">
                <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/>
                </svg>
              </div>
              <h2 className="text-3xl font-bold text-black mb-2">Upload Successful!</h2>
              <p className="text-gray-700">Your data has been processed and loaded into Snowflake</p>
            </div>

            {/* Stats Grid */}
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

            {/* Data Preview */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h3 className="text-xl font-bold mb-4 text-black">Data Preview (First 5 Rows)</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-100">
                    <tr>
                      {result.preview.columns.map((col, idx) => (
                        <th key={idx} className="px-4 py-3 text-left font-semibold text-gray-700 border-b-2 border-gray-300">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.preview.data.map((row, rowIdx) => (
                      <tr key={rowIdx} className="border-b border-gray-200 hover:bg-gray-50 transition">
                        {row.map((cell, cellIdx) => (
                          <td key={cellIdx} className="px-4 py-3 text-gray-800">
                            {cell !== null ? cell.toString() : 'NULL'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Info Box */}
              <div className="mt-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
                <p className="text-sm text-gray-700">
                  <strong>Next Steps:</strong> Your data is now ready for analysis. The workflow has been created 
                  and your CSV data is stored in Snowflake table <code className="bg-gray-200 px-2 py-1 rounded font-mono text-xs">{result.upload.table_name}</code>.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

    </div>
  );
}

export default Home;
