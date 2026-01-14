import { useState } from 'react'
import Sidebar from './components/Sidebar'
import './App.css'

function App() {
  const [inputText, setInputText] = useState('')
  const [summary, setSummary] = useState('')
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stats, setStats] = useState(null)

  const handleNewChat = () => {
    setInputText('')
    setSummary('')
    setStats(null)
    setProgress(0)
  }

  const handleExampleClick = (text) => {
    setInputText(text)
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
        body: JSON.stringify({ text: inputText }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let summaryText = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
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
                // Calculate stats
                const inputWords = inputText.trim().split(/\s+/).length
                const summaryWords = summaryText.trim().split(/\s+/).length
                const wordsReduced = inputWords - summaryWords
                const compressionRatio = (summaryWords / inputWords) * 100
                const aiFlag = Math.max(0, Math.min(100, 100 - compressionRatio + (summaryWords < 50 ? 20 : 0)))

                setStats({
                  inputWords,
                  summaryWords,
                  wordsReduced,
                  aiFlag: Math.round(aiFlag),
                })
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

  return (
    <div className="app">
      <Sidebar 
        onNewChat={handleNewChat} 
        chatHistory={[]} 
        onExampleClick={handleExampleClick}
      />
      
      <main className="main-content">
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
                  {stats && (
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