import React, { useMemo, useState } from "react";
import Footer from "../components/Footer";

export default function Help() {
  const [searchQuery, setSearchQuery] = useState("");
  const [categoryKey, setCategoryKey] = useState(null);

  // ✅ 20 Q&As
  const helpData = [
    {
      question: "How do I upload a CSV file?",
      answer:
        "Go to the ForeSee page and click 'Upload CSV'. Choose a CSV under 10MB. Once uploaded, you’ll see a preview and can proceed to analysis.",
      tags: ["upload", "csv", "file", "start", "getting started"],
    },
    {
      question: "What file types does Foresee support?",
      answer:
        "Right now, Foresee supports .csv files. We’re exploring support for spreadsheets in future updates.",
      tags: ["file types", "csv", "format", "getting started"],
    },
    {
      question: "My CSV failed to upload. What should I check?",
      answer:
        "Confirm the file is under 10MB, encoded in UTF-8, and that your first row contains headers. Remove empty trailing columns/rows and try again.",
      tags: ["upload", "error", "csv", "troubleshoot"],
    },
    {
      question: "How should my CSV be formatted?",
      answer:
        "Include a header row, avoid merged cells, ensure consistent column types, and use standard date formats (YYYY-MM-DD or ISO).",
      tags: ["csv", "format", "schema", "getting started"],
    },
    {
      question: "How do predictions work in Foresee?",
      answer:
        "Foresee detects the task (classification or regression), selects a suitable model, and evaluates it using cross-validation before generating predictions.",
      tags: ["model", "predictions", "how it works", "analysis"],
    },
    {
      question: "Can I choose which column to predict?",
      answer:
        "Yes. After uploading, select your target column from the dropdown. Foresee will use the remaining columns as features.",
      tags: ["target", "predict", "setup", "model"],
    },
    {
      question: "How does Foresee handle missing values?",
      answer:
        "Foresee applies safe imputations by default (median for numeric, mode for categorical). You’ll see a summary of what was imputed.",
      tags: ["missing values", "cleaning", "preprocessing", "model"],
    },
    {
      question: "How do I interpret the prediction results?",
      answer:
        "Use the Results panel: for regression, check MAE/RMSE; for classification, look at accuracy/F1. Feature importance explains key drivers.",
      tags: ["results", "metrics", "interpretation", "predictions"],
    },
    {
      question: "How accurate are the models?",
      answer:
        "Accuracy varies by data quality and size. We show metrics via cross-validation to provide a realistic estimate of generalization.",
      tags: ["accuracy", "metrics", "model", "predictions"],
    },
    {
      question: "Can I download the predictions?",
      answer:
        "Yes. Click 'Download Results' to export a CSV with predicted values and (optionally) probabilities or confidence intervals.",
      tags: ["download", "export", "results", "predictions"],
    },
    {
      question: "Is my data stored?",
      answer:
        "No. Your dataset is processed in-session and deleted afterward. We don’t retain your files or results unless you export them.",
      tags: ["privacy", "security", "storage", "data"],
    },
    {
      question: "What’s the maximum file size?",
      answer:
        "Up to 10MB per CSV. If your file is larger, consider sampling rows or reducing unused columns before upload.",
      tags: ["limits", "size", "upload", "getting started"],
    },
    {
      question: "I see weird characters after upload. Why?",
      answer:
        "This is often an encoding issue. Re-save the CSV with UTF-8 encoding and ensure the delimiter is a comma, then re-upload.",
      tags: ["encoding", "utf-8", "delimiter", "troubleshoot"],
    },
    {
      question: "Why did I get a 'no target column' warning?",
      answer:
        "Foresee couldn’t infer what to predict. Pick the target column manually in the setup step and run analysis again.",
      tags: ["target", "warning", "setup", "troubleshoot"],
    },
    {
      question: "How do I improve model performance?",
      answer:
        "Ensure clean data, remove outliers, add meaningful features, and provide enough rows (hundreds+ is ideal).",
      tags: ["performance", "quality", "features", "model"],
    },
    {
      question: "Can I balance imbalanced classes?",
      answer:
        "Foresee applies internal strategies. You’ll see class distribution in the report; consider gathering more minority class data if needed.",
      tags: ["classification", "imbalance", "model", "predictions"],
    },
    {
      question: "How can I view feature importance?",
      answer:
        "After analysis, open the 'Insights' panel. You’ll see top features and their relative contribution to the predictions.",
      tags: ["feature importance", "insights", "explainability", "model"],
    },
    {
      question: "Does Foresee support time series forecasting?",
      answer:
        "Basic forecasting works if a time column is detected. Ensure your dates are clean and sorted. Advanced features are coming soon.",
      tags: ["time series", "forecast", "dates", "model"],
    },
    {
      question: "Can I rerun analysis with different settings?",
      answer:
        "Yes. Adjust the target column or toggle options and click 'Re-run'.",
      tags: ["rerun", "settings", "analysis", "workflow"],
    },
    {
      question: "How do I contact support?",
      answer:
        "Email support@foresee.us or use the Contact Support link below for assistance.",
      tags: ["support", "contact", "help"],
    },
  ];

  const categoryTagMap = {
    "getting-started": [
      "getting started",
      "start",
      "upload",
      "csv",
      "file",
      "format",
      "limits",
      "download",
    ],
    model: [
      "model",
      "predictions",
      "accuracy",
      "metrics",
      "target",
      "missing values",
      "feature importance",
      "time series",
      "analysis",
    ],
  };

  const normalized = (s) => (s || "").toLowerCase();

  const resultsFromSearch = useMemo(() => {
    const q = normalized(searchQuery);
    if (!q) return [];
    return helpData.filter((item) => {
      const inQ = normalized(item.question).includes(q);
      const inA = normalized(item.answer).includes(q);
      const inTags = (item.tags || [])
        .map(normalized)
        .some((t) => t.includes(q));
      return inQ || inA || inTags;
    });
  }, [searchQuery, helpData]);

  const resultsFromCategory = useMemo(() => {
    if (!categoryKey) return [];
    const tagSet = new Set(categoryTagMap[categoryKey].map(normalized));
    return helpData.filter((item) =>
      (item.tags || []).some((t) => tagSet.has(normalized(t)))
    );
  }, [categoryKey, helpData]);

  const filteredResults =
    (categoryKey && resultsFromCategory.length > 0 && resultsFromCategory) ||
    (resultsFromSearch.length > 0 && resultsFromSearch) ||
    [];

  const filterByCategory = (key) => setCategoryKey(key);

  const onInput = (e) => {
    setSearchQuery(e.target.value);
    if (categoryKey) setCategoryKey(null);
  };

  return (
    <div className="min-h-screen font-poppins bg-[#EEEDE9] text-black pt-40">
      <div className="px-10 md:px-24 fade-up fade-delay-1">

        {/* ✅ Title */}
        <section className="text-center mb-8">
          <h1 className="text-3xl font-semibold mb-2">Help Center</h1>
          <p className="text-gray-700">
            Get answers, learn how ForeSee works and troubleshoot
          </p>
        </section>

        {/* ✅ Search Bar */}
        <div className="flex justify-center mb-12">
          <div className="w-full max-w-2xl flex items-center bg-white rounded-full shadow px-5 py-3">
            <input
              type="text"
              value={searchQuery}
              onChange={onInput}
              placeholder="Search help topics (e.g., csv, accuracy, download)…"
              className="w-full bg-transparent outline-none text-sm"
            />
            <span className="text-gray-500 text-xl">🔍</span>
          </div>
        </div>

        {/* ✅ Topic Cards (hide if searching or filtering) */}
        {!searchQuery && !categoryKey && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10 fade-up fade-delay-2">
            <div className="bg-white rounded-lg shadow p-6 flex flex-col justify-between">
              <div>
                <h3 className="font-semibold mb-2">Getting Started</h3>
                <p className="text-sm text-gray-700">
                  Learn how ForeSee works, how to upload files, and how to download your analysis.
                </p>
              </div>
              <div className="mt-4">
                <button
                  onClick={() => filterByCategory("getting-started")}
                  className="text-sm text-gray-900 font-medium hover:underline flex items-center gap-1"
                >
                  View answers →
                </button>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6 flex flex-col justify-between">
              <div>
                <h3 className="font-semibold mb-2">Model & Predictions</h3>
                <p className="text-sm text-gray-700">
                  Understand how ForeSee analyzes data and evaluates accuracy.
                </p>
              </div>
              <div className="mt-4">
                <button
                  onClick={() => filterByCategory("model")}
                  className="text-sm text-gray-900 font-medium hover:underline flex items-center gap-1"
                >
                  View answers →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ✅ Suggested Answers + No Results Message */}
        {(searchQuery || categoryKey) && (
          <section className="fade-up fade-delay-3 mb-16">
            {filteredResults.length > 0 ? (
              <>
                <h2 className="text-xl font-bold mb-6 text-center">Suggested Answers</h2>
                <div className="space-y-6 max-w-4xl mx-auto text-sm">
                  {filteredResults.map((item, idx) => (
                    <div
                      key={`${item.question}-${idx}`}
                      className="bg-white rounded-lg shadow p-5"
                    >
                      <p className="font-semibold mb-1">Q: {item.question}</p>
                      <p className="text-gray-700">A: {item.answer}</p>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="text-center text-sm text-gray-700">
                <p className="font-medium mb-2">No results found.</p>
                <p>
                  For complex questions, try{" "}
                  <a href="mailto:support@foresee.us" className="underline">
                    contacting support
                  </a>.
                </p>
              </div>
            )}
          </section>
        )}

        {/* ✅ FAQs (always visible) */}
        <section className="fade-up fade-delay-3 mb-16">
          <h2 className="text-xl font-bold mb-6 text-center">FAQs</h2>
          <div className="space-y-8 max-w-4xl mx-auto text-sm">
            <div>
              <p className="font-semibold">
                Q: Do I need to know programming or machine learning to use Foresee?
              </p>
              <p className="text-gray-700">
                Nope! Foresee was made for anyone who wants insights from data. Just upload your CSV and follow the prompts.
              </p>
            </div>

            <div>
              <p className="font-semibold">Q: What file types can I upload?</p>
              <p className="text-gray-700">
                Foresee currently supports .csv files up to 10MB.
              </p>
            </div>

            <div>
              <p className="font-semibold">Q: Does Foresee store my uploaded data?</p>
              <p className="text-gray-700">
                No. Your data is only used during processing and is deleted afterward.
              </p>
            </div>

            <div>
              <p className="font-semibold">Q: Can I re-run analysis with a different target?</p>
              <p className="text-gray-700">
                Yes. Select a different target column on the ForeSee page and click “Re-run”.
              </p>
            </div>
          </div>
        </section>

        {/* ✅ Contact */}
        <section className="fade-up fade-delay-4 mb-20 text-center">
          <h2 className="text-lg font-semibold mb-4">Contact</h2>
          <p className="text-sm">Didn’t find what you were looking for?</p>
          <p className="text-sm mt-2">
            <a href="mailto:support@foresee.us" className="underline">
              Contact Support
            </a>{" "}
            or email us at <strong>support@foresee.us</strong>
          </p>
        </section>
      </div>

      {/* ✅ Footer */}
      <Footer />
    </div>
  );
}
