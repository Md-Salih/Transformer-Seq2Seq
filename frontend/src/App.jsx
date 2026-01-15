import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import './App.css'

function App() {
  const [inputText, setInputText] = useState('')
  const [summary, setSummary] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stats, setStats] = useState(null)
  const [summaryLength, setSummaryLength] = useState(150)
  const [showStats, setShowStats] = useState(true)
  const [paraphrasing, setParaphrasing] = useState(false)
  const [sliderMode, setSliderMode] = useState('custom') // 'custom' or 'auto'
  const [sidebarVisible, setSidebarVisible] = useState(true)
  const [chatHistory, setChatHistory] = useState([])

  // Load chat history from localStorage on mount
  useEffect(() => {
    const loadHistory = () => {
      const stored = localStorage.getItem('chatHistory')
      if (stored) {
        const history = JSON.parse(stored)
        // Filter out entries from previous days
        const today = new Date().toDateString()
        const todayHistory = history.filter(item => {
          const itemDate = new Date(item.timestamp).toDateString()
          return itemDate === today
        })
        setChatHistory(todayHistory)
        // Update localStorage with filtered history
        localStorage.setItem('chatHistory', JSON.stringify(todayHistory))
      }
    }
    loadHistory()
  }, [])

  // Save chat to history
  const saveToHistory = (inputText, summary) => {
    // Generate title from first few words of summary (more meaningful)
    const words = summary.trim().split(/\s+/)
    const titleWords = words.slice(0, 5).join(' ') // First 5 words
    const title = titleWords + (words.length > 5 ? '...' : '')
    
    const newEntry = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      input: inputText,
      summary: summary,
      title: title || 'Untitled Summary'
    }
    const updatedHistory = [newEntry, ...chatHistory]
    setChatHistory(updatedHistory)
    localStorage.setItem('chatHistory', JSON.stringify(updatedHistory))
  }

  // Clear history at end of day
  useEffect(() => {
    const checkEndOfDay = () => {
      const now = new Date()
      const endOfDay = new Date()
      endOfDay.setHours(23, 59, 59, 999)
      const timeUntilMidnight = endOfDay - now
      
      if (timeUntilMidnight > 0) {
        setTimeout(() => {
          localStorage.removeItem('chatHistory')
          setChatHistory([])
        }, timeUntilMidnight)
      }
    }
    checkEndOfDay()
  }, [])

  const handleNewChat = () => {
    setInputText('')
    setSummary('')
    setStats(null)
    setProgress(0)
    setShowStats(true)
  }

  const handleExampleClick = (text) => {
    setInputText(text)
  }

  const handleParaphrase = async () => {
    if (!summary || paraphrasing) return
    
    setParaphrasing(true)
    try {
      const response = await fetch('/api/summarize/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          text: summary,
          max_length: summaryLength 
        }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let paraphrasedText = ''
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6))
            if (data.token) {
              paraphrasedText += data.token
              setSummary(paraphrasedText)
            }
          }
        }
      }
    } catch (error) {
      console.error('Paraphrase error:', error)
      alert('Failed to paraphrase summary. Please try again.')
    } finally {
      setParaphrasing(false)
    }
  }

  const handleToggleStats = () => {
    setShowStats(!showStats)
  }

  const handleDownload = () => {
    if (!summary) return
    
    const blob = new Blob([summary], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'summary.txt'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleCopy = async () => {
    if (!summary) return
    
    try {
      await navigator.clipboard.writeText(summary)
      // Optional: Show a toast notification
      const btn = document.querySelector('.icon-btn[title="Copy to Clipboard"]')
      if (btn) {
        const originalTitle = btn.title
        btn.title = 'Copied!'
        setTimeout(() => {
          btn.title = originalTitle
        }, 2000)
      }
    } catch (error) {
      console.error('Copy failed:', error)
      alert('Failed to copy to clipboard')
    }
  }

  const handleSubmit = async () => {
    if (!inputText.trim() || loading) return

    setLoading(true)
    setProgress(0)
    setSummary('')
    setStats(null)

    try {
      const response = await fetch('/api/summarize/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          text: inputText,
          max_length: summaryLength 
        }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let summaryText = ''
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // SSE events are separated by a blank line (\n\n)
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''

        for (const event of events) {
          const lines = event.split('\n')
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue

            try {
              const data = JSON.parse(line.slice(6))

              if (data.error) {
                throw new Error(data.error)
              }

              if (data.token) {
                summaryText += data.token
                setProgress(data.progress || 0)
                setSummary(summaryText)
              }

              if (data.done) {
                const inputWords = inputText.trim().split(/\s+/).length
                const summaryWords = summaryText.trim().split(/\s+/).length
                const wordsReduced = inputWords - summaryWords
                const compressionRatio = (summaryWords / inputWords) * 100

                // Heuristic fallback (kept for backward compatibility)
                const heuristicAiFlag = Math.max(0, Math.min(100, 100 - compressionRatio + (summaryWords < 50 ? 20 : 0)))

                setStats({
                  inputWords,
                  summaryWords,
                  wordsReduced,
                  aiFlag: Math.round(heuristicAiFlag),
                })

                // Save to history after successful summarization
                saveToHistory(inputText, summaryText)

                // Optional real detector (if backend is configured)
                try {
                  const detectRes = await fetch('/api/ai-detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: summaryText }),
                  })

                  if (detectRes.ok) {
                    const detectData = await detectRes.json()
                    if (detectData?.available && typeof detectData.ai_percent === 'number') {
                      setStats((prev) => prev ? ({ ...prev, aiFlag: detectData.ai_percent }) : prev)
                    }
                  }
                } catch {
                  // Ignore and keep heuristic
                }
              }
            } catch (parseError) {
              console.error('Parse error:', parseError)
            }
          }
        }
      }

      setLoading(false)
    } catch (error) {
      console.error('Error:', error)
      alert(`Error: ${error.message}. Please make sure the backend server is running on port 5000.`)
      setLoading(false)
    }
  }

  const handleSelectChat = (index) => {
    const selectedChat = chatHistory[index]
    if (selectedChat) {
      setInputText(selectedChat.input)
      setSummary(selectedChat.summary)
      // Recalculate stats
      const inputWords = selectedChat.input.trim().split(/\s+/).length
      const summaryWords = selectedChat.summary.trim().split(/\s+/).length
      setStats({
        inputWords,
        summaryWords,
        wordsReduced: inputWords - summaryWords,
        aiFlag: 0 // Reset AI flag for historical data
      })
    }
  }

  return (
    <div className="app">
      <Sidebar 
        onNewChat={handleNewChat} 
        chatHistory={chatHistory} 
        onExampleClick={handleExampleClick}
        isVisible={sidebarVisible}
        onToggle={() => setSidebarVisible(!sidebarVisible)}
        onSelectChat={handleSelectChat}
      />
      
      <main className={`main-content ${!sidebarVisible ? 'sidebar-hidden' : ''}`}>
        <div className="split-container">
          {/* Input Panel */}
          <div className="input-panel">
            <div className="panel-header">
              <h2 className="panel-title">Input Text</h2>
            </div>
            <div className="panel-content">
              <textarea
                className="input-textarea-large"
                placeholder="Enter or paste your text here to summarize..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
              />
            </div>
            <div className="panel-footer">
              <div className="footer-left">
                <button 
                  className="submit-btn"
                  onClick={handleSubmit}
                  disabled={loading || !inputText.trim()}
                >
                  {loading ? 'Generating...' : 'Summarize'}
                </button>
                <button 
                  className="clear-btn"
                  onClick={handleNewChat}
                >
                  Clear
                </button>
              </div>
              <div className="footer-right">
                <div className="slider-container-inline">
                  <div className="mode-toggle">
                    <button 
                      className={`mode-btn ${sliderMode === 'custom' ? 'active' : ''}`}
                      onClick={() => setSliderMode('custom')}
                    >
                      Custom
                    </button>
                    <button 
                      className={`mode-btn ${sliderMode === 'auto' ? 'active' : ''}`}
                      onClick={() => {
                        setSliderMode('auto')
                        setSummaryLength(150) // Auto mode uses default
                      }}
                    >
                      Auto
                    </button>
                  </div>
                  <div className="slider-labels-inline">
                    <span className="slider-label">BRIEF</span>
                    <span className="slider-value-display">{summaryLength} words</span>
                    <span className="slider-label">DETAILED</span>
                  </div>
                  <input 
                    type="range" 
                    min="50" 
                    max="300" 
                    step="10"
                    value={summaryLength}
                    onChange={(e) => setSummaryLength(Number(e.target.value))}
                    className="summary-slider-inline"
                    disabled={sliderMode === 'auto'}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Output Panel */}
          <div className="output-panel">
            <div className="panel-header">
              <h2 className="panel-title">Summary</h2>
            </div>
            <div className="panel-content">
              {loading && (
                <div>
                  <div className="loading-text">Generating summary... {progress}%</div>
                  <div className="line-loader-container">
                    <div className="line-loader" style={{ width: `${progress}%` }}></div>
                  </div>
                </div>
              )}
              {!loading && !summary && (
                <div className="output-empty">
                  Your summary will appear here
                </div>
              )}
              {summary && (
                <>
                  <div className="output-text">{summary}</div>
                  {stats && showStats && (
                    <div className="stats-grid">
                      <div className="stat-box">
                        <div className="stat-value">{stats.inputWords}</div>
                        <div className="stat-label">Input Words</div>
                      </div>
                      <div className="stat-box">
                        <div className="stat-value">{stats.summaryWords}</div>
                        <div className="stat-label">Summary Words</div>
                      </div>
                      <div className="stat-box">
                        <div className="stat-value">{stats.wordsReduced}</div>
                        <div className="stat-label">Words Reduced</div>
                      </div>
                      <div className="stat-box">
                        <div className="stat-value">{stats.aiFlag}%</div>
                        <div className="stat-label">AI Detection</div>
                      </div>
                    </div>
                  )}
                  <div className="summary-actions">
                    <button 
                      className="paraphrase-btn"
                      onClick={handleParaphrase}
                      disabled={paraphrasing}
                    >
                      {paraphrasing ? 'Paraphrasing...' : 'Paraphrase Summary'}
                    </button>
                    <div className="action-icons">
                      <button 
                        className={`icon-btn ${showStats ? 'active' : ''}`}
                        onClick={handleToggleStats}
                        title={showStats ? "Hide Statistics" : "Show Statistics"}
                      >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <rect x="3" y="3" width="7" height="7" strokeWidth="2"/>
                          <rect x="14" y="3" width="7" height="7" strokeWidth="2"/>
                          <rect x="14" y="14" width="7" height="7" strokeWidth="2"/>
                          <rect x="3" y="14" width="7" height="7" strokeWidth="2"/>
                        </svg>
                      </button>
                      <button 
                        className="icon-btn"
                        onClick={handleDownload}
                        title="Download Summary"
                      >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <polyline points="7 10 12 15 17 10" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <line x1="12" y1="15" x2="12" y2="3" strokeWidth="2" strokeLinecap="round"/>
                        </svg>
                      </button>
                      <button 
                        className="icon-btn"
                        onClick={handleCopy}
                        title="Copy to Clipboard"
                      >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App