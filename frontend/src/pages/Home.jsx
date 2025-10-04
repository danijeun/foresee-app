function App() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: '#EEEDE9' }}>
      {/* ✅ Navbar */}
      <header className="w-full flex justify-between items-center px-8 py-4 shadow-sm bg-white">
        <div className="text-2xl font-bold" style={{ color: '#3D57A9' }}>
          FOREsee
        </div>
        <nav className="space-x-6" style={{ color: '#3D57A9' }}>
          <a href="#" className="hover:opacity-70">Foresee</a>
          <a href="#" className="hover:opacity-70">Need Help?</a>
          <a href="#" className="hover:opacity-70">About Us</a>
        </nav>
      </header>

      {/* ✅ Hero Section */}
      <section className="text-center py-20 px-4">
        <h1
          className="text-4xl md:text-5xl font-extrabold mb-4"
          style={{ color: '#3D57A9' }}
        >
          Analysis Made Easy
        </h1>
        <p className="text-lg text-gray-700 mb-12 max-w-2xl mx-auto">
          Drag and drop a CSV file and let AI generate insights, predictions,
          and reports instantly.
        </p>

        {/* ✅ Upload Box */}
        <div className="max-w-xl mx-auto border-2 border-dashed rounded-lg p-10 bg-white"
          style={{ borderColor: '#3D57A9' }}
        >
          <div className="mb-4 text-sm" style={{ color: '#3D57A9' }}>
            Drag & drop CSV here or click to select
          </div>
          <div className="text-6xl mb-4" style={{ color: '#3D57A9' }}>📤</div>
          <button
            className="px-6 py-2 rounded-md transition text-white"
            style={{ backgroundColor: '#3D57A9' }}
          >
            Upload CSV
          </button>
        </div>
      </section>
    </div>
  );
}

export default App;
