import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const welcomeMessage =
  "Paste a text, then ask about words, meaning, grammar, or usage.";

const initialMessages = [
  {
    role: "assistant",
    content: welcomeMessage,
  },
];

const jlptLabels = {
  1: "N1",
  2: "N2",
  3: "N3",
  4: "N4",
  5: "N5",
};

const targetLevelOptions = ["N5", "N4", "N3", "N2", "N1"];

const japaneseWordPattern = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffー々〆〤]/;
const japanesePunctuationPattern = /^[。、！？「」『』（）［］【】・…ー\s]+$/;
function clampScore(score) {
  return Math.max(0, Math.min(100, score));
}

function canLookupToken(token) {
  return (
    japaneseWordPattern.test(token.text) &&
    !japanesePunctuationPattern.test(token.text)
  );
}

function average(scores) {
  return scores.length
    ? scores.reduce((sum, score) => sum + score, 0) / scores.length
    : 0;
}

function percentile(scores, percentileValue) {
  if (scores.length === 0) return 0;

  const sortedScores = [...scores].sort((a, b) => a - b);
  const index = Math.ceil((percentileValue / 100) * sortedScores.length) - 1;
  return sortedScores[Math.max(0, Math.min(sortedScores.length - 1, index))];
}

function levelToScore(level) {
  return clampScore(((level - 1) / 5) * 100);
}

function getVocabularyLevel(token) {
  if (token.jlpt_level) return 6 - token.jlpt_level;
  if (!token.frequency_rank) return 6;
  if (token.frequency_rank <= 2000) return 3;
  if (token.frequency_rank <= 5000) return 4;
  if (token.frequency_rank <= 10000) return 5;
  return 6;
}

function getKanjiLevel(token) {
  const knownLevels = (token.kanji_levels ?? []).map((level) => 6 - level);
  const unknownLevels = Array.from(
    { length: token.unknown_kanji_count ?? 0 },
    () => 6,
  );
  const levels = [...knownLevels, ...unknownLevels];

  if (levels.length === 0) return null;
  return average(levels) * 0.55 + Math.max(...levels) * 0.45;
}

function getGrammarScore(text) {
  const sentenceCount = Math.max(
    1,
    text.split(/[。！？!?]+/).filter((sentence) => sentence.trim()).length,
  );
  const classicalMatches = text.match(/べき|ざる|しめる|ぬ|をば|とて/g) ?? [];
  const voiceMatches =
    text.match(/(?:させられる|せられる|させる|される|られる|れる)/g) ?? [];
  const grammarHitsPerSentence =
    (classicalMatches.length * 1.4 + voiceMatches.length) / sentenceCount;

  return clampScore(grammarHitsPerSentence * 38);
}

function getSentenceLengthScore(annotations) {
  const sentenceLengths = [];
  let currentLength = 0;

  annotations.forEach((token) => {
    if (canLookupToken(token)) currentLength += 1;
    if (/[。！？!?]/.test(token.text)) {
      if (currentLength > 0) sentenceLengths.push(currentLength);
      currentLength = 0;
    }
  });
  if (currentLength > 0) sentenceLengths.push(currentLength);

  const meanSentenceLength = average(sentenceLengths);
  return clampScore(((meanSentenceLength - 6) / 24) * 100);
}

function getTextStats(text, annotations) {
  const distribution = [1, 2, 3, 4, 5].reduce(
    (levels, level) => ({ ...levels, [level]: 0 }),
    {},
  );
  const lookupTokens = annotations.filter((token) => canLookupToken(token));
  const contentTokens = lookupTokens.filter(
    (token) => token.is_content && !token.is_proper_noun,
  );

  contentTokens.forEach((token) => {
    if (token.jlpt_level) {
      distribution[token.jlpt_level] += 1;
    }
  });

  const vocabularyLevels = contentTokens.map(getVocabularyLevel);
  const vocabularyLevelScore = vocabularyLevels.length
    ? average(
        [
          average(vocabularyLevels),
          percentile(vocabularyLevels, 75),
          percentile(vocabularyLevels, 90),
        ].map(levelToScore),
      )
    : 0;
  const contentTypes = new Set(
    contentTokens.map((token) => token.base_form ?? token.text),
  );
  const typeTokenRatio = contentTokens.length
    ? contentTypes.size / contentTokens.length
    : 0;
  const vocabScore = clampScore(
    vocabularyLevelScore + Math.max(0, typeTokenRatio - 0.45) * 18,
  );

  const kanjiCharacters = [...text].filter((char) =>
    /[\u3400-\u9fff\uf900-\ufaff]/.test(char),
  ).length;
  const nonSpaceCharacters = [...text].filter(
    (char) => !/\s/.test(char),
  ).length;
  const kanjiDensity = kanjiCharacters / Math.max(1, nonSpaceCharacters);
  const kanjiLevels = lookupTokens
    .map(getKanjiLevel)
    .filter((level) => level !== null);
  const kanjiLevelScore = kanjiLevels.length
    ? average(
        [average(kanjiLevels), percentile(kanjiLevels, 75)].map(levelToScore),
      )
    : 0;
  const kanjiScore = clampScore(
    kanjiLevelScore * 0.7 + kanjiDensity * 100 * 0.3,
  );

  const sentenceLengthScore = getSentenceLengthScore(annotations);
  const grammarScore = getGrammarScore(text);
  const difficultyScore = Math.round(
    clampScore(
      vocabScore * 0.5 +
        kanjiScore * 0.2 +
        sentenceLengthScore * 0.15 +
        grammarScore * 0.15,
    ),
  );

  const jlptWordCount = Object.values(distribution).reduce(
    (sum, count) => sum + count,
    0,
  );
  const frequencyWordCount = contentTokens.filter(
    (token) => token.frequency_rank,
  ).length;

  return {
    characters: text.length,
    charactersNoSpaces: nonSpaceCharacters,
    lines: text.length === 0 ? 0 : text.split("\n").length,
    tokens: annotations.filter((token) => token.text.trim() !== "").length,
    lookupTokens: lookupTokens.length,
    jlptWordCount,
    frequencyWordCount,
    jlptDistribution: distribution,
    difficultyScore,
  };
}

function getDifficultyLabel(score) {
  if (score >= 86) return "Advanced";
  if (score >= 66) return "Upper-intermediate";
  if (score >= 40) return "Intermediate";
  if (score > 0) return "Foundational";
  return "Awaiting JLPT words";
}

function getDefaultEasierTarget(distribution) {
  const dominantLevel = [1, 2, 3, 4, 5].reduce((currentBest, level) =>
    distribution[level] > distribution[currentBest] ? level : currentBest,
  );

  if (!distribution[dominantLevel]) return "N4";
  return jlptLabels[Math.min(5, dominantLevel + 1)];
}

function App() {
  const [text, setText] = useState("");
  const [messages, setMessages] = useState(initialMessages);
  const [chatInput, setChatInput] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  const [isAnnotating, setIsAnnotating] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [activeTextTab, setActiveTextTab] = useState("source");
  const [targetLevel, setTargetLevel] = useState("N4");
  const [easierText, setEasierText] = useState("");
  const [easierTextSource, setEasierTextSource] = useState("");
  const [easierAnnotations, setEasierAnnotations] = useState([]);
  const [pendingSimplifyLevel, setPendingSimplifyLevel] = useState(null);
  const [isSimplifying, setIsSimplifying] = useState(false);
  const [annotations, setAnnotations] = useState([]);
  const [translations, setTranslations] = useState({});
  const [activeTooltip, setActiveTooltip] = useState(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const editorRef = useRef(null);
  const easierEditorRef = useRef(null);
  const pendingTranslationsRef = useRef(new Set());

  const activeText = activeTextTab === "easier" ? easierText : text;
  const activeAnnotations = activeTextTab === "easier" ? easierAnnotations : annotations;
  const sourceStats = useMemo(
    () => getTextStats(text, annotations),
    [text, annotations],
  );
  const stats = useMemo(
    () => getTextStats(activeText, activeAnnotations),
    [activeText, activeAnnotations],
  );
  const maxJlptCount = Math.max(1, ...Object.values(stats.jlptDistribution));
  const difficultyLabel = getDifficultyLabel(stats.difficultyScore);
  const defaultEasierTarget = useMemo(
    () => getDefaultEasierTarget(sourceStats.jlptDistribution),
    [sourceStats.jlptDistribution],
  );
  const sourceNeedsTokenization = japaneseWordPattern.test(text) && annotations.length === 0;
  const isSourceTokenized = text.trim() !== "" && !isAnnotating && !sourceNeedsTokenization;

  useEffect(() => {
    setMessages((currentMessages) =>
      currentMessages.map((message) =>
        message.content.includes("こんにちは") ||
        message.content.includes("JLPT")
          ? { ...message, content: welcomeMessage }
          : message,
      ),
    );
  }, []);

  useEffect(() => {
    setTargetLevel(defaultEasierTarget);
  }, [defaultEasierTarget]);

  useEffect(() => {
    setEasierText("");
    setEasierTextSource("");
    setEasierAnnotations([]);
    setPendingSimplifyLevel(null);
  }, [text]);

  useEffect(() => {
    if (!pendingSimplifyLevel || !isSourceTokenized || isSimplifying) return;

    const level = pendingSimplifyLevel;
    setPendingSimplifyLevel(null);
    generateEasierText(level, { skipSourceTokenizationCheck: true });
    // generateEasierText is intentionally omitted: this effect is a small gate
    // that releases one queued simplify request only after source tokenization finishes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSimplifying, isSourceTokenized, pendingSimplifyLevel]);

  useEffect(() => {
    const trimmedText = text.trim();
    if (!trimmedText) {
      setAnnotations([]);
      setIsAnnotating(false);
      return undefined;
    }

    setIsAnnotating(true);
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/annotate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
          signal: controller.signal,
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail ?? "Could not annotate Japanese text.");
        }
        setAnnotations(data.tokens ?? []);
      } catch (annotationError) {
        if (annotationError.name !== "AbortError") {
          setError(annotationError.message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsAnnotating(false);
        }
      }
    }, 350);

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [text]);

  async function loadTranslation(word) {
    if (
      !word.trim() ||
      translations[word] ||
      pendingTranslationsRef.current.has(word)
    )
      return;

    pendingTranslationsRef.current.add(word);
    setTranslations((currentTranslations) => ({
      ...currentTranslations,
      [word]: "Looking up…",
    }));

    try {
      const response = await fetch(`${API_BASE_URL}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Translation unavailable");
      }
      setTranslations((currentTranslations) => ({
        ...currentTranslations,
        [word]: data.translation || "Translation unavailable",
      }));
      if (data.reading) {
        const addReading = (currentAnnotations) =>
          currentAnnotations.map((token) =>
            (token.text === word || token.base_form === word) && !token.reading
              ? { ...token, reading: data.reading }
              : token,
          );
        setAnnotations(addReading);
        setEasierAnnotations(addReading);
        setActiveTooltip((currentTooltip) =>
          currentTooltip?.lookupWord === word && !currentTooltip.reading
            ? { ...currentTooltip, reading: data.reading }
            : currentTooltip,
        );
      }
    } catch {
      setTranslations((currentTranslations) => ({
        ...currentTranslations,
        [word]: "Translation unavailable",
      }));
    } finally {
      pendingTranslationsRef.current.delete(word);
    }
  }

  function showTooltip(event, token) {
    const rect = event.currentTarget.getBoundingClientRect();
    const lookupWord = token.base_form ?? token.text;
    setActiveTooltip({
      word: token.text,
      lookupWord,
      baseForm: token.base_form,
      level: token.jlpt_level,
      reading: token.reading,
      left: rect.left + rect.width / 2,
      top: rect.top - 10,
    });
    loadTranslation(lookupWord);
  }

  function handleEditorInput(event) {
    setAnnotations([]);
    setText(event.currentTarget.innerText);
  }

  function renderAnnotatedText(editor, tokenList, fallbackText) {
    if (!editor) return;

    editor.replaceChildren();
    const tokens = tokenList.length > 0 ? tokenList : fallbackText ? [{ text: fallbackText }] : [];

    tokens.forEach((token, index) => {
      const level = token.jlpt_level;
      const hasLevel = level !== null && level !== undefined;
      const canLookup = canLookupToken(token);
      const span = document.createElement("span");
      span.textContent = token.text;
      span.className = hasLevel
        ? `annotated-token jlpt-n${level}`
        : canLookup
          ? "lookup-token"
          : "plain-token";
      if (canLookup) {
        span.tabIndex = 0;
        span.dataset.tokenIndex = String(index);
      }
      editor.appendChild(span);
    });
  }

  useEffect(() => {
    renderAnnotatedText(editorRef.current, annotations, text);
  }, [activeTextTab, annotations, text]);

  useEffect(() => {
    if (isSimplifying) return;
    renderAnnotatedText(easierEditorRef.current, easierAnnotations, easierText);
  }, [activeTextTab, easierAnnotations, easierText, isSimplifying]);

  function handleEditorPointer(event) {
    const tokenElement = event.target.closest?.("[data-token-index]");
    if (!tokenElement || !event.currentTarget.contains(tokenElement)) {
      setActiveTooltip(null);
      return;
    }

    const token = activeAnnotations[Number(tokenElement.dataset.tokenIndex)];
    if (token) {
      showTooltip({ currentTarget: tokenElement }, token);
    } else {
      setActiveTooltip(null);
    }
  }

  function handleEditorPaste(event) {
    event.preventDefault();
    const pastedText = event.clipboardData.getData("text/plain");
    document.execCommand("insertText", false, pastedText);
  }

  async function handleFileUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setError("");
    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Upload failed");
      }

      setText(data.content);
      setUploadedFileName(data.filename);
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setIsUploading(false);
      event.target.value = "";
    }
  }

  async function generateEasierText(level = targetLevel, options = {}) {
    const sourceText = text.trim();
    if (!sourceText || isSimplifying) return;

    if (!options.skipSourceTokenizationCheck && !isSourceTokenized) {
      setPendingSimplifyLevel(level);
      setActiveTextTab("easier");
      return;
    }

    setError("");
    setIsSimplifying(true);

    setEasierText("");
    setEasierAnnotations([]);

    try {
      const response = await fetch(`${API_BASE_URL}/simplify/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sourceText, target_level: level }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail ?? "Could not create an easier version.");
      }
      if (!response.body) {
        throw new Error("Streaming is not available in this browser.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);

          if (event.type === "chunk") {
            setEasierText((currentText) => `${currentText}${event.text ?? ""}`);
            await new Promise((resolve) => window.requestAnimationFrame(resolve));
          } else if (event.type === "done") {
            setEasierText(event.text ?? "");
            setEasierAnnotations(event.annotations ?? []);
            setEasierTextSource(sourceText);
          } else if (event.type === "error") {
            throw new Error(event.detail ?? "Could not create an easier version.");
          }
        }
      }
    } catch (simplifyError) {
      setError(simplifyError.message);
    } finally {
      setIsSimplifying(false);
    }
  }

  function handleTextTabChange(tab) {
    setActiveTextTab(tab);
    if (tab === "easier" && text.trim() && !easierText) {
      generateEasierText();
    }
  }

  function handleTargetLevelChange(event) {
    const nextLevel = event.target.value;
    setTargetLevel(nextLevel);
    setPendingSimplifyLevel(null);
    if (activeTextTab === "easier" && text.trim()) {
      generateEasierText(nextLevel);
    }
  }

  async function handleSendMessage(event) {
    event.preventDefault();

    const message = chatInput.trim();
    if (!message || isAnswering) return;

    const userMessage = { role: "user", content: message };
    setMessages((currentMessages) => [...currentMessages, userMessage]);
    setChatInput("");
    setError("");
    setIsAnswering(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "The assistant could not answer.");
      }

      setMessages((currentMessages) => [
        ...currentMessages,
        { role: "assistant", content: data.answer },
      ]);
    } catch (chatError) {
      setError(chatError.message);
      setMessages((currentMessages) => [
        ...currentMessages,
        {
          role: "assistant",
          content: `Sorry, I could not get an answer. ${chatError.message}`,
        },
      ]);
    } finally {
      setIsAnswering(false);
    }
  }

  function clearText() {
    setText("");
    setUploadedFileName("");
    setActiveTextTab("source");
    setEasierText("");
    setEasierTextSource("");
    setEasierAnnotations([]);
    setPendingSimplifyLevel(null);
  }

  function clearChat() {
    setMessages(initialMessages);
    setError("");
  }

  return (
    <main className="app-shell">
      <nav className="top-nav" aria-label="App navigation">
        <div className="brand-mark">
          <span>Language Assistant</span>
          <strong>Readr</strong>
        </div>
        <div className="nav-actions">
          {uploadedFileName && <span className="nav-file-name">{uploadedFileName}</span>}
          <button
            className="nav-button"
            type="button"
            onClick={() => setShowInfo((isVisible) => !isVisible)}
            aria-expanded={showInfo}
          >
            Info
          </button>
          <input
            id="text-upload"
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.csv,text/plain,text/markdown,text/csv"
            onChange={handleFileUpload}
          />
          <button
            className="nav-button nav-upload"
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            {isUploading ? "Uploading…" : "Upload text"}
          </button>
        </div>
      </nav>

      {showInfo && (
        <aside className="info-strip" role="note">
          Paste or upload Japanese text, inspect JLPT/frequency difficulty, ask the tutor questions, and generate an easier AI rewrite for your target level.
        </aside>
      )}

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
              <article
                key={`${message.role}-${index}`}
                className={`message ${message.role}`}
              >
                <span className="message-role">
                  {message.role === "assistant" ? "Tutor" : "You"}
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
            <button
              type="submit"
              disabled={isAnswering || chatInput.trim() === ""}
            >
              Send
            </button>
          </form>
        </aside>

        <section className="panel text-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Study text</p>
              <h2>{activeTextTab === "source" ? "Paste or edit study text" : "AI easier version"}</h2>
            </div>
            <div className="text-actions">
              {isAnnotating && activeTextTab === "source" && (
                <span className="reader-status">Tokenizing…</span>
              )}
              {isSimplifying && activeTextTab === "easier" && (
                <span className="reader-status">Rewriting…</span>
              )}
              <button
                className="ghost-button"
                type="button"
                onClick={clearText}
              >
                Clear text
              </button>
            </div>
          </div>

          <div className="text-tabs" role="tablist" aria-label="Text versions">
            <button
              className={activeTextTab === "source" ? "is-active" : ""}
              type="button"
              role="tab"
              aria-selected={activeTextTab === "source"}
              onClick={() => handleTextTabChange("source")}
            >
              Source
            </button>
            <button
              className={activeTextTab === "easier" ? "is-active" : ""}
              type="button"
              role="tab"
              aria-selected={activeTextTab === "easier"}
              onClick={() => handleTextTabChange("easier")}
            >
              Easier
            </button>
          </div>

          <div className="jlpt-legend" aria-label="JLPT color legend">
            {[5, 4, 3, 2, 1].map((level) => (
              <span key={level} className={`legend-chip jlpt-n${level}`}>
                {jlptLabels[level]}
              </span>
            ))}
          </div>

          {activeTextTab === "source" ? (
            <>
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
            </>
          ) : (
            <div className="easier-pane">
              <div className="easier-controls">
                <label htmlFor="target-level">Target difficulty</label>
                <select
                  id="target-level"
                  value={targetLevel}
                  onChange={handleTargetLevelChange}
                  disabled={isSimplifying || !text.trim()}
                >
                  {targetLevelOptions.map((level) => (
                    <option key={level} value={level}>
                      {level}
                    </option>
                  ))}
                </select>
                <button
                  className="secondary-button"
                  type="button"
                  onClick={() => generateEasierText()}
                  disabled={isSimplifying || Boolean(pendingSimplifyLevel) || !text.trim()}
                >
                  {pendingSimplifyLevel
                    ? "Tokenizing…"
                    : isSimplifying
                      ? "Generating…"
                      : easierText
                        ? "Regenerate"
                        : "Generate"}
                </button>
              </div>

              {!easierText ? (
                <div className="easier-output empty-output" lang="ja">
                  <p className="empty-state">
                    {pendingSimplifyLevel
                      ? "Tokenizing the source text first…"
                      : isSimplifying
                        ? `Starting a ${targetLevel} version…`
                        : "Paste source text, then open this tab to generate a simpler version."}
                  </p>
                </div>
              ) : isSimplifying ? (
                <div className="easier-output empty-output streaming-output" lang="ja">
                  <p>{easierText}</p>
                </div>
              ) : (
                <div className="editor-frame easier-output" lang="ja">
                  <div
                    ref={easierEditorRef}
                    className="study-editor"
                    onBlur={() => setActiveTooltip(null)}
                    onFocus={handleEditorPointer}
                    onMouseEnter={handleEditorPointer}
                    onMouseLeave={() => setActiveTooltip(null)}
                    onMouseMove={handleEditorPointer}
                    role="document"
                  />
                </div>
              )}
              {easierText && easierTextSource !== text.trim() && (
                <p className="easier-note">
                  Source text changed. Regenerate to refresh this version.
                </p>
              )}
            </div>
          )}
          {activeTooltip && (
            <div
              className={`token-popover is-visible ${activeTooltip.level ? `jlpt-n${activeTooltip.level}` : "dictionary-token"}`}
              role="tooltip"
              style={{ left: activeTooltip.left, top: activeTooltip.top }}
            >
              <strong>
                {activeTooltip.word}
                {activeTooltip.baseForm &&
                  activeTooltip.baseForm !== activeTooltip.word && (
                    <em className="popover-reading">
                      → {activeTooltip.baseForm}
                    </em>
                  )}
                {activeTooltip.reading &&
                  activeTooltip.reading !== activeTooltip.word && (
                    <em className="popover-reading">{activeTooltip.reading}</em>
                  )}
                {activeTooltip.level && (
                  <span className="popover-level">
                    {jlptLabels[activeTooltip.level]}
                  </span>
                )}
              </strong>
              <span>
                {translations[activeTooltip.lookupWord] ?? "Looking up…"}
              </span>
            </div>
          )}
        </section>

        <aside className="panel stats-panel">
          <div>
            <p className="eyebrow">Ledger</p>
            <h2>Difficulty map</h2>
          </div>

          <section
            className="difficulty-card"
            aria-label="Text difficulty score"
          >
            <span className="difficulty-label">{difficultyLabel}</span>
            <strong>{stats.difficultyScore}</strong>
            <small>
              Vocabulary percentiles plus kanji density, sentence length, and
              grammar signals
            </small>
          </section>

          <section className="jlpt-chart" aria-label="JLPT word distribution">
            <div className="chart-header">
              <span>JLPT / frequency content words</span>
              <strong>
                {stats.jlptWordCount} / {stats.frequencyWordCount}
              </strong>
            </div>
            {[1, 2, 3, 4, 5].map((level) => {
              const count = stats.jlptDistribution[level];
              const width = count
                ? `${Math.max(4, (count / maxJlptCount) * 100)}%`
                : "0%";
              return (
                <div className="chart-row" key={level}>
                  <span className={`chart-level jlpt-n${level}`}>
                    {jlptLabels[level]}
                  </span>
                  <div className="chart-track" aria-hidden="true">
                    <span
                      className={`chart-bar jlpt-n${level}`}
                      style={{ width }}
                    />
                  </div>
                  <strong>{count}</strong>
                </div>
              );
            })}
          </section>

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
  );
}

export default App;
