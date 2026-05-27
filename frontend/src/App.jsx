import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

const welcomeMessage = 'Paste a text, then ask about words, meaning, grammar, or usage.'

const initialMessages = [
  {
    role: 'assistant',
    content: welcomeMessage,
  },
]

const jlptLabels = {
  1: 'N1',
  2: 'N2',
  3: 'N3',
  4: 'N4',
  5: 'N5',
}

const japaneseWordPattern = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffー々〆〤]/
const japanesePunctuationPattern = /^[。、！？「」『』（）［］【】・…ー\s]+$/

function canLookupToken(token) {
  return japaneseWordPattern.test(token.text) && !japanesePunctuationPattern.test(token.text)
}

function getTextStats(text, annotations) {
  return {
    characters: text.length,
    charactersNoSpaces: [...text].filter((char) => !/\s/.test(char)).length,
    lines: text.length === 0 ? 0 : text.split('\n').length,
    tokens: annotations.filter((token) => token.text.trim() !== '').length,
  }
}

function App() {
  const [text, setText] = useState('')
  const [messages, setMessages] = useState(initialMessages)
  const [chatInput, setChatInput] = useState('')
  const [uploadedFileName, setUploadedFileName] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [isAnswering, setIsAnswering] = useState(false)
  const [isAnnotating, setIsAnnotating] = useState(false)
  const [annotations, setAnnotations] = useState([])
  const [translations, setTranslations] = useState({})
  const [activeTooltip, setActiveTooltip] = useState(null)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)
  const editorRef = useRef(null)
  const pendingTranslationsRef = useRef(new Set())

  const stats = useMemo(() => getTextStats(text, annotations), [text, annotations])

  useEffect(() => {
    setMessages((currentMessages) =>
      currentMessages.map((message) =>
        message.content.includes('こんにちは') || message.content.includes('JLPT')
          ? { ...message, content: welcomeMessage }
          : message,
      ),
    )
  }, [])

  useEffect(() => {
    const trimmedText = text.trim()
    if (!trimmedText) {
      setAnnotations([])
      return undefined
    }

    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      setIsAnnotating(true)
      try {
        const response = await fetch(`${API_BASE_URL}/annotate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
          signal: controller.signal,
        })
        const data = await response.json()
        if (!response.ok) {
          throw new Error(data.detail ?? 'Could not annotate Japanese text.')
        }
        setAnnotations(data.tokens ?? [])
      } catch (annotationError) {
        if (annotationError.name !== 'AbortError') {
          setError(annotationError.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsAnnotating(false)
        }
      }
    }, 350)

    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [text])

  async function loadTranslation(word) {
    if (!word.trim() || translations[word] || pendingTranslationsRef.current.has(word)) return

    pendingTranslationsRef.current.add(word)
    setTranslations((currentTranslations) => ({
      ...currentTranslations,
      [word]: 'Looking up…',
    }))

    try {
      const response = await fetch(`${API_BASE_URL}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word }),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail ?? 'Translation unavailable')
      }
      setTranslations((currentTranslations) => ({
        ...currentTranslations,
        [word]: data.translation || 'Translation unavailable',
      }))
      if (data.reading) {
        setAnnotations((currentAnnotations) =>
          currentAnnotations.map((token) =>
            token.text === word && !token.reading ? { ...token, reading: data.reading } : token,
          ),
        )
        setActiveTooltip((currentTooltip) =>
          currentTooltip?.word === word && !currentTooltip.reading
            ? { ...currentTooltip, reading: data.reading }
            : currentTooltip,
        )
      }
    } catch {
      setTranslations((currentTranslations) => ({
        ...currentTranslations,
        [word]: 'Translation unavailable',
      }))
    } finally {
      pendingTranslationsRef.current.delete(word)
    }
  }

  function showTooltip(event, token) {
    const rect = event.currentTarget.getBoundingClientRect()
    setActiveTooltip({
      word: token.text,
      level: token.jlpt_level,
      reading: token.reading,
      left: rect.left + rect.width / 2,
      top: rect.top - 10,
    })
    loadTranslation(token.text)
  }

  function handleEditorInput(event) {
    setAnnotations([])
    setText(event.currentTarget.innerText)
  }

  useEffect(() => {
    const editor = editorRef.current
    if (!editor) return

    editor.replaceChildren()
    const tokens = annotations.length > 0 ? annotations : text ? [{ text }] : []

    tokens.forEach((token, index) => {
      const level = token.jlpt_level
      const hasLevel = level !== null && level !== undefined
      const canLookup = canLookupToken(token)
      const span = document.createElement('span')
      span.textContent = token.text
      span.className = hasLevel
        ? `annotated-token jlpt-n${level}`
        : canLookup
          ? 'lookup-token'
          : 'plain-token'
      if (canLookup) {
        span.tabIndex = 0
        span.dataset.tokenIndex = String(index)
      }
      editor.appendChild(span)
    })
  }, [annotations, text])

  function handleEditorPointer(event) {
    const tokenElement = event.target.closest?.('[data-token-index]')
    if (!tokenElement || !event.currentTarget.contains(tokenElement)) {
      setActiveTooltip(null)
      return
    }

    const token = annotations[Number(tokenElement.dataset.tokenIndex)]
    if (token) {
      showTooltip({ currentTarget: tokenElement }, token)
    } else {
      setActiveTooltip(null)
    }
  }

  function handleEditorPaste(event) {
    event.preventDefault()
    const pastedText = event.clipboardData.getData('text/plain')
    document.execCommand('insertText', false, pastedText)
  }

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
          <h1>Readr</h1>
          <p className="subtitle">
            A quiet study desk for any language text: keep the source material in front
            of you, track the basics, and ask targeted questions as you work.
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
              <p className="eyebrow">Margin notes</p>
              <h2>Ask the tutor</h2>
            </div>
            <button className="ghost-button" type="button" onClick={clearChat}>
              Clear
            </button>
          </div>

          <div className="messages" aria-live="polite">
            {messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
                <span className="message-role">
                  {message.role === 'assistant' ? 'Tutor' : 'You'}
                </span>
                <p>{message.content}</p>
              </article>
            ))}
            {isAnswering && (
              <article className="message assistant">
                <span className="message-role">Tutor</span>
                <p>Thinking…</p>
              </article>
            )}
          </div>

          <form className="chat-form" onSubmit={handleSendMessage}>
            <input
              type="text"
              value={chatInput}
              onChange={(event) => setChatInput(event.target.value)}
              placeholder="Ask about meaning, grammar, usage, or difficulty…"
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
              <p className="eyebrow">Source text</p>
              <h2>Paste or edit study text</h2>
            </div>
            <div className="text-actions">
              {isAnnotating && <span className="reader-status">Tokenizing…</span>}
              <button className="ghost-button" type="button" onClick={clearText}>
                Clear text
              </button>
            </div>
          </div>
          <div className="jlpt-legend" aria-label="JLPT color legend">
            {[5, 4, 3, 2, 1].map((level) => (
              <span key={level} className={`legend-chip jlpt-n${level}`}>
                {jlptLabels[level]}
              </span>
            ))}
          </div>
          <div className="editor-frame">
            <div
              ref={editorRef}
              className="study-editor"
              contentEditable
              data-placeholder="Paste text here, or upload a .txt, .md, or .csv file above…"
              lang="ja"
              onBlur={() => setActiveTooltip(null)}
              onFocus={handleEditorPointer}
              onInput={handleEditorInput}
              onMouseEnter={handleEditorPointer}
              onMouseLeave={() => setActiveTooltip(null)}
              onMouseMove={handleEditorPointer}
              onPaste={handleEditorPaste}
              role="textbox"
              spellCheck="false"
              suppressContentEditableWarning
            />
          </div>
          {activeTooltip && (
            <div
              className={`token-popover is-visible ${activeTooltip.level ? `jlpt-n${activeTooltip.level}` : 'dictionary-token'}`}
              role="tooltip"
              style={{ left: activeTooltip.left, top: activeTooltip.top }}
            >
              <strong>
                {activeTooltip.word}
                {activeTooltip.reading && activeTooltip.reading !== activeTooltip.word && (
                  <em className="popover-reading">{activeTooltip.reading}</em>
                )}
                {activeTooltip.level && (
                  <span className="popover-level">{jlptLabels[activeTooltip.level]}</span>
                )}
              </strong>
              <span>{translations[activeTooltip.word] ?? 'Looking up…'}</span>
            </div>
          )}
        </section>

        <aside className="panel stats-panel">
          <div>
            <p className="eyebrow">Ledger</p>
            <h2>Text count</h2>
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
              <span>Tokens</span>
              <strong>{stats.tokens}</strong>
            </div>
          </div>
        </aside>
      </section>
    </main>
  )
}

export default App
