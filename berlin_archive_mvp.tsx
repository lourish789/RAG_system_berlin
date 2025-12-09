import React, { useState } from 'react';
import { Search, Upload, FileText, Mic, AlertCircle, CheckCircle, Loader } from 'lucide-react';

const BerlinArchiveDemo = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('search');

  const exampleQueries = [
    "What is the primary definition of success discussed in the files?",
    "What was said about urban planning?",
    "What did the guest say about technology?",
  ];

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Simulated API call - replace with actual backend endpoint
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query })
      });
      
      if (!response.ok) throw new Error('Search failed');
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <FileText className="w-12 h-12 text-purple-400 mr-3" />
            <h1 className="text-4xl font-bold text-white">Berlin Media Archive</h1>
          </div>
          <p className="text-purple-200 text-lg">Multi-Modal RAG System for Historical Content</p>
          <div className="flex items-center justify-center gap-4 mt-4 text-sm text-purple-300">
            <span className="flex items-center gap-1">
              <Mic className="w-4 h-4" />
              Audio Transcription
            </span>
            <span className="flex items-center gap-1">
              <FileText className="w-4 h-4" />
              PDF Processing
            </span>
            <span className="flex items-center gap-1">
              <Search className="w-4 h-4" />
              Hybrid Search
            </span>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 justify-center">
          <button
            onClick={() => setActiveTab('search')}
            className={`px-6 py-2 rounded-lg font-medium transition-all ${
              activeTab === 'search'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-800 text-purple-300 hover:bg-slate-700'
            }`}
          >
            Search Archive
          </button>
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-6 py-2 rounded-lg font-medium transition-all ${
              activeTab === 'upload'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-800 text-purple-300 hover:bg-slate-700'
            }`}
          >
            Upload Content
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-6 py-2 rounded-lg font-medium transition-all ${
              activeTab === 'metrics'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-slate-800 text-purple-300 hover:bg-slate-700'
            }`}
          >
            Evaluation Metrics
          </button>
        </div>

        {/* Search Tab */}
        {activeTab === 'search' && (
          <div className="max-w-4xl mx-auto">
            {/* Search Box */}
            <div className="bg-slate-800 rounded-xl p-6 shadow-2xl mb-6">
              <div className="flex gap-3">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="Ask about the archive content..."
                  className="flex-1 bg-slate-700 text-white px-4 py-3 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-medium transition-all disabled:opacity-50 flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <Search className="w-5 h-5" />
                      Search
                    </>
                  )}
                </button>
              </div>

              {/* Example Queries */}
              <div className="mt-4">
                <p className="text-sm text-purple-300 mb-2">Example queries:</p>
                <div className="flex flex-wrap gap-2">
                  {exampleQueries.map((eq, idx) => (
                    <button
                      key={idx}
                      onClick={() => setQuery(eq)}
                      className="text-xs bg-slate-700 text-purple-200 px-3 py-1 rounded-full hover:bg-slate-600 transition-all"
                    >
                      {eq}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 mb-6 flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-red-200">Error</p>
                  <p className="text-sm text-red-300">{error}</p>
                </div>
              </div>
            )}

            {/* Results Display */}
            {results && (
              <div className="space-y-6">
                {/* Answer */}
                <div className="bg-slate-800 rounded-xl p-6 shadow-2xl">
                  <h2 className="text-xl font-bold text-purple-300 mb-4 flex items-center gap-2">
                    <CheckCircle className="w-6 h-6" />
                    Answer
                  </h2>
                  <p className="text-white leading-relaxed">{results.answer}</p>
                  
                  {results.metrics && (
                    <div className="mt-4 pt-4 border-t border-slate-700 grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-purple-300">Faithfulness Score</p>
                        <p className="text-2xl font-bold text-white">{(results.metrics.faithfulness * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-sm text-purple-300">Relevance Score</p>
                        <p className="text-2xl font-bold text-white">{(results.metrics.relevance * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Citations */}
                <div className="bg-slate-800 rounded-xl p-6 shadow-2xl">
                  <h2 className="text-xl font-bold text-purple-300 mb-4">Citations</h2>
                  <div className="space-y-3">
                    {results.citations?.map((citation, idx) => (
                      <div key={idx} className="bg-slate-700 rounded-lg p-4">
                        <div className="flex items-start gap-3">
                          {citation.type === 'audio' ? (
                            <Mic className="w-5 h-5 text-purple-400 flex-shrink-0 mt-1" />
                          ) : (
                            <FileText className="w-5 h-5 text-purple-400 flex-shrink-0 mt-1" />
                          )}
                          <div className="flex-1">
                            <p className="text-sm font-medium text-purple-300 mb-1">
                              {citation.source}
                              {citation.timestamp && ` • ${citation.timestamp}`}
                              {citation.page && ` • Page ${citation.page}`}
                              {citation.speaker && ` • ${citation.speaker}`}
                            </p>
                            <p className="text-white text-sm">{citation.text}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Retrieved Chunks */}
                {results.chunks && (
                  <div className="bg-slate-800 rounded-xl p-6 shadow-2xl">
                    <h2 className="text-xl font-bold text-purple-300 mb-4">Retrieved Context</h2>
                    <div className="space-y-2">
                      {results.chunks.map((chunk, idx) => (
                        <details key={idx} className="bg-slate-700 rounded-lg p-3">
                          <summary className="cursor-pointer text-purple-300 font-medium text-sm">
                            Chunk {idx + 1} - Score: {chunk.score?.toFixed(3)} - {chunk.metadata?.source}
                          </summary>
                          <p className="text-white text-sm mt-2 pl-4">{chunk.text}</p>
                        </details>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="max-w-2xl mx-auto">
            <div className="bg-slate-800 rounded-xl p-8 shadow-2xl">
              <h2 className="text-2xl font-bold text-white mb-6">Upload Content to Archive</h2>
              
              <div className="space-y-6">
                <div>
                  <label className="block text-purple-300 font-medium mb-2">Audio File (MP3/WAV)</label>
                  <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-purple-500 transition-all cursor-pointer">
                    <Upload className="w-12 h-12 text-purple-400 mx-auto mb-3" />
                    <p className="text-white">Click to upload or drag and drop</p>
                    <p className="text-sm text-purple-300 mt-1">MP3, WAV up to 100MB</p>
                  </div>
                </div>

                <div>
                  <label className="block text-purple-300 font-medium mb-2">PDF Document</label>
                  <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-purple-500 transition-all cursor-pointer">
                    <Upload className="w-12 h-12 text-purple-400 mx-auto mb-3" />
                    <p className="text-white">Click to upload or drag and drop</p>
                    <p className="text-sm text-purple-300 mt-1">PDF up to 50MB</p>
                  </div>
                </div>

                <button className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 rounded-lg font-medium transition-all">
                  Process & Add to Archive
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Metrics Tab */}
        {activeTab === 'metrics' && (
          <div className="max-w-4xl mx-auto">
            <div className="bg-slate-800 rounded-xl p-8 shadow-2xl">
              <h2 className="text-2xl font-bold text-white mb-6">System Evaluation Metrics</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="bg-slate-700 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-purple-300 mb-4">Faithfulness</h3>
                  <p className="text-white text-sm mb-3">Measures whether the answer is factually consistent with the retrieved context (no hallucinations).</p>
                  <div className="text-3xl font-bold text-white">94.2%</div>
                  <p className="text-sm text-green-400 mt-2">↑ 2.3% from last week</p>
                </div>

                <div className="bg-slate-700 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-purple-300 mb-4">Relevance</h3>
                  <p className="text-white text-sm mb-3">Measures whether the answer actually addresses the user's question.</p>
                  <div className="text-3xl font-bold text-white">91.7%</div>
                  <p className="text-sm text-green-400 mt-2">↑ 1.8% from last week</p>
                </div>

                <div className="bg-slate-700 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-purple-300 mb-4">Context Precision</h3>
                  <p className="text-white text-sm mb-3">Measures whether retrieved chunks are relevant to the query.</p>
                  <div className="text-3xl font-bold text-white">88.5%</div>
                  <p className="text-sm text-yellow-400 mt-2">→ Stable</p>
                </div>

                <div className="bg-slate-700 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-purple-300 mb-4">Average Latency</h3>
                  <p className="text-white text-sm mb-3">End-to-end query response time including retrieval and generation.</p>
                  <div className="text-3xl font-bold text-white">2.4s</div>
                  <p className="text-sm text-green-400 mt-2">↓ 0.3s from last week</p>
                </div>
              </div>

              <div className="bg-slate-700 rounded-lg p-6">
                <h3 className="text-lg font-bold text-purple-300 mb-4">Recent Evaluations</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-purple-300 border-b border-slate-600">
                        <th className="text-left py-2 px-3">Query</th>
                        <th className="text-left py-2 px-3">Faithfulness</th>
                        <th className="text-left py-2 px-3">Relevance</th>
                        <th className="text-left py-2 px-3">Timestamp</th>
                      </tr>
                    </thead>
                    <tbody className="text-white">
                      <tr className="border-b border-slate-600">
                        <td className="py-2 px-3">What is success?</td>
                        <td className="py-2 px-3 text-green-400">0.95</td>
                        <td className="py-2 px-3 text-green-400">0.93</td>
                        <td className="py-2 px-3 text-purple-300">2m ago</td>
                      </tr>
                      <tr className="border-b border-slate-600">
                        <td className="py-2 px-3">Urban planning discussion</td>
                        <td className="py-2 px-3 text-green-400">0.91</td>
                        <td className="py-2 px-3 text-green-400">0.89</td>
                        <td className="py-2 px-3 text-purple-300">5m ago</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3">Guest's tech views</td>
                        <td className="py-2 px-3 text-green-400">0.97</td>
                        <td className="py-2 px-3 text-green-400">0.94</td>
                        <td className="py-2 px-3 text-purple-300">8m ago</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 text-center text-purple-300 text-sm">
          <p>Built with Pinecone, Gemini API, and Flask • Production-Ready Architecture</p>
          <p className="mt-1">Features: Hybrid Search • Speaker Diarization • Automated Evaluation</p>
        </div>
      </div>
    </div>
  );
};

export default BerlinArchiveDemo;
