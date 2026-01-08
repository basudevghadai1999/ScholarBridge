import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { BookOpen, Sparkles, ArrowRight, Loader2, CheckCircle, FileText } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

function App() {
    const [url, setUrl] = useState('')
    const [mode, setMode] = useState('deep') // 'fast' or 'deep'
    const [status, setStatus] = useState('idle') // idle, loading, complete, error
    const [logs, setLogs] = useState([])
    const [result, setResult] = useState(null)

    const eventSourceRef = useRef(null)

    const startAnalysis = async () => {
        if (!url) return
        setStatus('loading')
        setLogs([])
        setResult(null)

        // For this V1, we will use a simple POST and poll/wait, 
        // but to support streaming logs we'd need SSE. 
        // Given the constraints, let's use a standard fetch for the final result 
        // and maybe simulate logs or just show "Processing...".
        // EDIT: Let's assume the backend returns the JSON result directly for now.

        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, mode })
            })

            const data = await response.json()
            if (data.final_blog) {
                setResult(data)
                setStatus('complete')
            } else {
                setStatus('error')
            }
        } catch (e) {
            console.error(e)
            setStatus('error')
        }
    }

    return (
        <div className="min-h-screen bg-neutral-950 text-white font-outfit selection:bg-purple-500/30">

            {/* Navbar */}
            <nav className="border-b border-white/5 bg-black/50 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center">
                            <BookOpen size={18} className="text-white" />
                        </div>
                        <span className="font-bold text-xl tracking-tight">ScholarBridge</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                        <div className="flex bg-neutral-900 rounded-lg p-1 border border-white/10">
                            <button
                                onClick={() => setMode('fast')}
                                className={`px-3 py-1 rounded-md transition ${mode === 'fast' ? 'bg-white text-black font-semibold' : 'text-neutral-400 hover:text-white'}`}
                            >
                                Fast (API)
                            </button>
                            <button
                                onClick={() => setMode('deep')}
                                className={`px-3 py-1 rounded-md transition ${mode === 'deep' ? 'bg-purple-600 text-white font-semibold' : 'text-neutral-400 hover:text-white'}`}
                            >
                                Deep (PDF)
                            </button>
                        </div>
                    </div>
                </div>
            </nav>

            <main className="max-w-7xl mx-auto px-6 py-12">

                {/* Hero Input */}
                <section className="max-w-3xl mx-auto text-center mb-16">
                    <h1 className="text-5xl font-bold mb-6 bg-gradient-to-r from-white via-neutral-200 to-neutral-500 bg-clip-text text-transparent">
                        Turn Research into Content.
                    </h1>
                    <p className="text-lg text-neutral-400 mb-8">
                        {mode === 'deep' ? "Deep Dive Mode: Downloads & reads full PDFs for maximum authority." : "Fast Mode: Quickly scans abstracts for rapid insights."}
                    </p>

                    <div className="relative group">
                        <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
                        <div className="relative flex gap-2 bg-neutral-900 p-2 rounded-xl border border-white/10 shadow-2xl">
                            <input
                                type="text"
                                value={url}
                                onChange={(e) => setUrl(e.target.value)}
                                placeholder="https://openai.com"
                                className="flex-1 bg-transparent border-none outline-none text-white px-4 placeholder:text-neutral-600"
                            />
                            <button
                                onClick={startAnalysis}
                                disabled={status === 'loading'}
                                className="bg-white text-black px-6 py-3 rounded-lg font-semibold hover:bg-neutral-200 transition flex items-center gap-2 disabled:opacity-50"
                            >
                                {status === 'loading' ? <Loader2 className="animate-spin" /> : <Sparkles size={18} />}
                                {status === 'loading' ? 'Running Agents...' : 'Generate'}
                            </button>
                        </div>
                    </div>
                </section>

                {/* Results Area */}
                <AnimatePresence>
                    {result && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="grid grid-cols-1 md:grid-cols-3 gap-8"
                        >
                            {/* Left: Metadata */}
                            <div className="space-y-6">
                                <div className="bg-neutral-900/50 border border-white/5 rounded-2xl p-6 backdrop-blur-sm">
                                    <h3 className="text-neutral-400 text-xs font-bold uppercase tracking-wider mb-4">Brand Analysis</h3>
                                    <div className="space-y-4">
                                        <div>
                                            <label className="text-sm text-neutral-500">Industry Niche</label>
                                            <div className="font-medium text-lg">{result.niche}</div>
                                        </div>
                                        <div>
                                            <label className="text-sm text-neutral-500">Brand Voice</label>
                                            <div className="font-medium text-lg">{result.brand_voice}</div>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-neutral-900/50 border border-white/5 rounded-2xl p-6 backdrop-blur-sm">
                                    <h3 className="text-neutral-400 text-xs font-bold uppercase tracking-wider mb-4">Selected Research</h3>
                                    {result.best_paper ? (
                                        <div>
                                            <div className="text-purple-400 text-xs font-bold mb-1">ARXIV PAPER</div>
                                            <a href={result.best_paper.entry_id} target="_blank" className="font-semibold hover:underline block mb-2 leading-snug">
                                                {result.best_paper.title}
                                            </a>
                                            <p className="text-xs text-neutral-400 line-clamp-3">
                                                {result.best_paper.abstract}
                                            </p>
                                        </div>
                                    ) : (
                                        <div className="text-neutral-500 italic">No relevant paper found.</div>
                                    )}
                                </div>
                            </div>

                            {/* Right: Blog Content */}
                            <div className="md:col-span-2">
                                <div className="bg-white text-neutral-900 rounded-2xl p-8 md:p-12 shadow-2xl relative overflow-hidden">
                                    <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-purple-500 via-blue-500 to-purple-500"></div>

                                    <div className="cms-content prose prose-lg prose-headings:font-display prose-p:font-outfit max-w-none">
                                        <ReactMarkdown>
                                            {result.final_blog}
                                        </ReactMarkdown>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

            </main>
        </div>
    )
}

export default App
