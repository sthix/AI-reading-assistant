import { useMemo, useRef, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

const initialMessages = [
  {
    role: 'assistant',
    content:
      'こんにちは! Paste or upload Japanese text, then ask me about vocabulary, readings, meanings, or JLPT levels.',
  },
]

function getTextStats(text) {
  return {
    characters: text.length,
    charactersNoSpaces: [...text].filter((char) => !/\s/.test(char)).length,
    lines: text.length === 0 ? 0 : text.split('\n').length,
    words: text.trim() === '' ? 0 : text.trim().split(/\s+/).length,
  }
}

function App() {
  const [text, setText] = useState('')
  const [messages, setMessages] = useState(initialMessages)
  const [chatInput, setChatInput] = useState('')
  const [uploadedFileName, setUploadedFileName] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [isAnswering, setIsAnswering] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)

  const stats = useMemo(() => getTextStats(text), [text])

  async function handleFileUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return

    setError('')
    setIsUploading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail ?? 'Upload failed')
      }

      setText(data.content)
      setUploadedFileName(data.filename)
    } catch (uploadError) {
      setError(uploadError.message)
    } finally {
      setIsUploading(false)
      event.target.value = ''
    }
  }

  async function handleSendMessage(event) {
    event.preventDefault()

    const message = chatInput.trim()
    if (!message || isAnswering) return

    const userMessage = { role: 'user', content: message }
    setMessages((currentMessages) => [...currentMessages, userMessage])
    setChatInput('')
    setError('')
    setIsAnswering(true)

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail ?? 'The assistant could not answer.')
      }

      setMessages((currentMessages) => [
        ...currentMessages,
        { role: 'assistant', content: data.answer },
      ])
    } catch (chatError) {
      setError(chatError.message)
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          role: 'assistant',
          content: `Sorry, I could not get an answer. ${chatError.message}`,
        },
      ])
    } finally {
      setIsAnswering(false)
    }
  }

  function clearText() {
    setText('')
    setUploadedFileName('')
  }

  function clearChat() {
    setMessages(initialMessages)
    setError('')
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Language Assistant</p>
          <h1>JLPT Vocabulary Assistant</h1>
          <p className="subtitle">
            Upload or paste Japanese text, inspect basic statistics, and chat with
            the RAG vocabulary tutor.
          </p>
        </div>
        <div className="upload-card">
          <label className="file-label" htmlFor="text-upload">
            Upload text file
          </label>
          <input
            id="text-upload"
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.csv,text/plain,text/markdown,text/csv"
            onChange={handleFileUpload}
          />
          <button
            className="secondary-button"
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            {isUploading ? 'Uploading…' : 'Choose file'}
          </button>
          {uploadedFileName && <p className="file-name">{uploadedFileName}</p>}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="workspace">
        <aside className="panel chat-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Tutor</p>
              <h2>Chat</h2>
            </div>
            <button className="ghost-button" type="button" onClick={clearChat}>
              Clear
            </button>
          </div>

          <div className="messages" aria-live="polite">
            {messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                <span className="message-role">
                  {message.role === 'assistant' ? 'Assistant' : 'You'}
                </span>
                <p>{message.content}</p>
              </article>
            ))}
            {isAnswering && (
              <article className="message assistant">
                <span className="message-role">Assistant</span>
                <p>Thinking…</p>
              </article>
            )}
          </div>

          <form className="chat-form" onSubmit={handleSendMessage}>
            <input
              type="text"
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              placeholder="Ask about meaning, reading, or JLPT level…"
              disabled={isAnswering}
            />
            <button type="submit" disabled={isAnswering || chatInput.trim() === ''}>
              Send
            </button>
          </form>
        </aside>

        <section className="panel text-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Reading text</p>
              <h2>Paste or edit text</h2>
            </div>
            <button className="ghost-button" type="button" onClick={clearText}>
              Clear text
            </button>
          </div>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Paste Japanese text here, or upload a .txt, .md, or .csv file above…"
            spellCheck="false"
          />
        </section>

        <aside className="panel stats-panel">
          <div>
            <p className="eyebrow">Overview</p>
            <h2>Statistics</h2>
          </div>
          <div className="stat-grid">
            <div className="stat-card">
              <span>Characters</span>
              <strong>{stats.characters}</strong>
            </div>
            <div className="stat-card">
              <span>No spaces</span>
              <strong>{stats.charactersNoSpaces}</strong>
            </div>
            <div className="stat-card">
              <span>Lines</span>
              <strong>{stats.lines}</strong>
            </div>
            <div className="stat-card">
              <span>Words</span>
              <strong>{stats.words}</strong>
            </div>
          </div>
        </aside>
      </section>
    </main>
  )
}

export default App
