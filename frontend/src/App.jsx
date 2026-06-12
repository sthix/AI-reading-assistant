import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const languageConfigs = {
  japanese: {
    label: "Japanese",
    shortLabel: "日本語",
    navLabel: "Japanese Assistant",
    tutorTitle: "Ask the Japanese tutor",
    welcome:
      "Paste Japanese text, then ask about words, meaning, JLPT level, grammar, or usage.",
    info:
      "Paste or upload Japanese text, inspect JLPT/frequency difficulty, ask the Japanese tutor questions, and generate an easier AI rewrite for your target level.",
    chatPlaceholder: "Ask about a Japanese word, meaning, usage, or JLPT level…",
    textTitle: "Paste or edit Japanese text",
    placeholder: "Paste Japanese text here, or upload a .txt, .md, or .csv file above…",
    lang: "ja",
    dir: "ltr",
    scaleLabel: "JLPT",
    listedLabel: "JLPT / frequency content words",
    levels: ["N5", "N4", "N3", "N2", "N1"],
    chartLevels: ["N5", "N4", "N3", "N2", "N1"],
    targetLevels: ["N5", "N4", "N3", "N2", "N1"],
    defaultTarget: "N4",
    classForLevel: (level) => `jlpt-n${level.replace("N", "")}`,
  },
  hebrew: {
    label: "Hebrew",
    shortLabel: "עברית",
    navLabel: "Hebrew Assistant",
    tutorTitle: "Ask the Hebrew tutor",
    welcome:
      "Paste Hebrew text, then ask about words, meaning, CEFR level, or usage.",
    info:
      "Paste or upload Hebrew text, inspect CEFR A1-C1 difficulty, ask the Hebrew tutor questions, and generate an easier AI rewrite for your target level.",
    chatPlaceholder: "Ask about a Hebrew word, meaning, usage, or CEFR level…",
    textTitle: "Paste or edit Hebrew text",
    placeholder: "Paste Hebrew text here, or upload a .txt, .md, or .csv file above…",
    lang: "he",
    dir: "rtl",
    scaleLabel: "CEFR",
    listedLabel: "CEFR / listed Hebrew words",
    levels: ["A1", "A2", "B1", "B2", "C1"],
    chartLevels: ["C1", "B2", "B1", "A2", "A1"],
    targetLevels: ["A1", "A2", "B1", "B2", "C1"],
    defaultTarget: "B1",
    classForLevel: (level) => `cefr-${level.toLowerCase()}`,
  },
};

const japaneseWordPattern = /[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffー々〆〤]/;
const japanesePunctuationPattern = /^[。、！？「」『』（）［］【】・…ー\s]+$/;
const hebrewWordPattern = /[\u0590-\u05ff]/;
const hebrewPunctuationPattern = /^[.,!?;:()[\]{}\s־״׳'"-]+$/;
const hebrewBidiControlPattern = /[\u200e\u200f\u202a-\u202e\u2066-\u2069]/g;
const hebrewTableRulePattern = /^\s*[|:;–—\s-]+\s*$/gm;

function getInitialMessages(language) {
  return [{ role: "assistant", content: languageConfigs[language].welcome }];
}

function normalizeSourceForLanguage(value, language) {
  const trimmed = value.trim();
  if (language !== "hebrew") return trimmed;
  return trimmed
    .replace(hebrewBidiControlPattern, "")
    .replace(/\r\n?/g, "\n")
    .replace(/(^|\s)\d+($|\s)/g, " ")
    .replace(hebrewTableRulePattern, " ")
    .replace(/[|]+/g, " ")
    .replace(/(?:^|\n)\s*[-*•]+\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function clampScore(score) {
  return Math.max(0, Math.min(100, score));
}

function canLookupToken(token, language) {
  if (language === "hebrew") {
    return (
      hebrewWordPattern.test(token.text) &&
      !hebrewPunctuationPattern.test(token.text)
    );
  }
  return (
    japaneseWordPattern.test(token.text) &&
    !japanesePunctuationPattern.test(token.text)
  );
}

function getHebrewSeparatorText(value) {
  const cleaned = value
    .replace(hebrewBidiControlPattern, "")
    .replace(/[|]+/g, " ")
    .replace(/\r\n?/g, "\n")
    .replace(/\s+/g, " ");

  if (!cleaned.trim()) return " ";
  return hebrewPunctuationPattern.test(cleaned) ? cleaned : " ";
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

function levelToScore(level, language) {
  if (language === "hebrew") {
    const index = languageConfigs.hebrew.levels.indexOf(level);
    if (index < 0) return null;
    return clampScore((index / (languageConfigs.hebrew.levels.length - 1)) * 100);
  }
  const number = Number(String(level).replace("N", ""));
  if (!number) return null;
  return clampScore(((5 - number) / 4) * 100);
}

function getJapaneseVocabularyLevel(token) {
  if (token.jlpt_level) return 6 - token.jlpt_level;
  if (!token.frequency_rank) return null;
  if (token.frequency_rank <= 2000) return 3;
  if (token.frequency_rank <= 5000) return 4;
  if (token.frequency_rank <= 10000) return 5;
  return 6;
}

function getJapaneseKanjiLevel(token) {
  const knownLevels = (token.kanji_levels ?? []).map((level) => 6 - level);
  const unknownLevels = Array.from(
    { length: token.unknown_kanji_count ?? 0 },
    () => 6,
  );
  const levels = [...knownLevels, ...unknownLevels];

  if (levels.length === 0) return null;
  return average(levels) * 0.55 + Math.max(...levels) * 0.45;
}

function getJapaneseGrammarScore(text) {
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

function getSentenceLengthScore(annotations, language) {
  const sentenceLengths = [];
  let currentLength = 0;

  annotations.forEach((token) => {
    if (canLookupToken(token, language)) currentLength += 1;
    const sentencePattern = language === "japanese" ? /[。！？!?]/ : /[.!?؟]+/;
    if (sentencePattern.test(token.text)) {
      if (currentLength > 0) sentenceLengths.push(currentLength);
      currentLength = 0;
    }
  });
  if (currentLength > 0) sentenceLengths.push(currentLength);

  const meanSentenceLength = average(sentenceLengths);
  const baseline = language === "japanese" ? 6 : 7;
  const spread = language === "japanese" ? 24 : 22;
  return clampScore(((meanSentenceLength - baseline) / spread) * 100);
}

function getHebrewVocabularyScore(token) {
  if (token.cefr_level) return levelToScore(token.cefr_level, "hebrew");
  if (!token.frequency_rank) return null;
  if (token.frequency_rank <= 250) return levelToScore("A1", "hebrew");
  if (token.frequency_rank <= 500) return levelToScore("A2", "hebrew");
  if (token.frequency_rank <= 800) return levelToScore("B1", "hebrew");
  if (token.frequency_rank <= 1000) return levelToScore("B2", "hebrew");
  return levelToScore("C1", "hebrew");
}

function getTextStats(text, annotations, language) {
  const config = languageConfigs[language];
  const distribution = config.levels.reduce(
    (levels, level) => ({ ...levels, [level]: 0 }),
    {},
  );
  const lookupTokens = annotations.filter((token) => canLookupToken(token, language));
  const contentTokens = lookupTokens.filter(
    (token) => token.is_content && !token.is_proper_noun,
  );

  if (language === "hebrew") {
    contentTokens.forEach((token) => {
      if (token.cefr_level && distribution[token.cefr_level] !== undefined) {
        distribution[token.cefr_level] += 1;
      }
    });

    const vocabularyScores = contentTokens
      .map(getHebrewVocabularyScore)
      .filter((score) => score !== null);
    const vocabularyLevelScore = vocabularyScores.length
      ? average([
          average(vocabularyScores),
          percentile(vocabularyScores, 75),
          percentile(vocabularyScores, 90),
        ])
      : 0;
    const contentTypes = new Set(contentTokens.map((token) => token.text));
    const typeTokenRatio = contentTokens.length
      ? contentTypes.size / contentTokens.length
      : 0;
    const typeTokenConfidence = Math.min(1, contentTokens.length / 40);
    const vocabScore = clampScore(
      vocabularyLevelScore +
        Math.max(0, typeTokenRatio - 0.5) * 18 * typeTokenConfidence,
    );
    const sentenceLengthScore = getSentenceLengthScore(annotations, language);
    const difficultyScore = Math.round(
      clampScore(vocabScore * 0.75 + sentenceLengthScore * 0.25),
    );
    const leveledWordCount = Object.values(distribution).reduce(
      (sum, count) => sum + count,
      0,
    );
    const listedWordCount = contentTokens.filter(
      (token) => token.cefr_level || token.frequency_rank,
    ).length;

    return {
      characters: text.length,
      charactersNoSpaces: [...text].filter((char) => !/\s/.test(char)).length,
      lines: text.length === 0 ? 0 : text.split("\n").length,
      tokens: annotations.filter((token) => token.text.trim() !== "").length,
      lookupTokens: lookupTokens.length,
      leveledWordCount,
      listedWordCount,
      distribution,
      difficultyScore,
    };
  }

  contentTokens.forEach((token) => {
    if (token.jlpt_level) {
      distribution[`N${token.jlpt_level}`] += 1;
    }
  });

  const vocabularyLevels = contentTokens
    .map(getJapaneseVocabularyLevel)
    .filter((level) => level !== null);
  const vocabularyLevelScore = vocabularyLevels.length
    ? average(
        [
          average(vocabularyLevels),
          percentile(vocabularyLevels, 75),
          percentile(vocabularyLevels, 90),
        ].map((level) => clampScore(((level - 1) / 5) * 100)),
      )
    : 0;
  const contentTypes = new Set(
    contentTokens.map((token) => token.base_form ?? token.text),
  );
  const typeTokenRatio = contentTokens.length
    ? contentTypes.size / contentTokens.length
    : 0;
  const typeTokenConfidence = Math.min(1, contentTokens.length / 40);
  const vocabScore = clampScore(
    vocabularyLevelScore +
      Math.max(0, typeTokenRatio - 0.45) * 18 * typeTokenConfidence,
  );

  const kanjiCharacters = [...text].filter((char) =>
    /[\u3400-\u9fff\uf900-\ufaff]/.test(char),
  ).length;
  const nonSpaceCharacters = [...text].filter(
    (char) => !/\s/.test(char),
  ).length;
  const kanjiDensity = kanjiCharacters / Math.max(1, nonSpaceCharacters);
  const kanjiLevels = lookupTokens
    .map(getJapaneseKanjiLevel)
    .filter((level) => level !== null);
  const kanjiLevelScore = kanjiLevels.length
    ? average(
        [average(kanjiLevels), percentile(kanjiLevels, 75)].map((level) =>
          clampScore(((level - 1) / 5) * 100),
        ),
      )
    : 0;
  const kanjiScore = clampScore(
    kanjiLevelScore * 0.7 + kanjiDensity * 100 * 0.3,
  );

  const sentenceLengthScore = getSentenceLengthScore(annotations, language);
  const grammarScore = getJapaneseGrammarScore(text);
  const difficultyScore = Math.round(
    clampScore(
      vocabScore * 0.5 +
        kanjiScore * 0.2 +
        sentenceLengthScore * 0.15 +
        grammarScore * 0.15,
    ),
  );
  const leveledWordCount = Object.values(distribution).reduce(
    (sum, count) => sum + count,
    0,
  );
  const listedWordCount = contentTokens.filter(
    (token) => token.frequency_rank,
  ).length;

  return {
    characters: text.length,
    charactersNoSpaces: nonSpaceCharacters,
    lines: text.length === 0 ? 0 : text.split("\n").length,
    tokens: annotations.filter((token) => token.text.trim() !== "").length,
    lookupTokens: lookupTokens.length,
    leveledWordCount,
    listedWordCount,
    distribution,
    difficultyScore,
  };
}

function getDifficultyLabel(score, language, context = {}) {
  if (language === "hebrew") {
    if (score >= 86) return "C1 advanced";
    if (score >= 66) return "B2 upper-intermediate";
    if (score >= 40) return "B1 intermediate";
    if (score > 0) return "A1–A2 foundational";
    if (context.hasListedWords) return "A1 foundational";
    if (context.isEasierTab && context.hasGeneratedText && !context.hasVisibleText) {
      return "No new Hebrew words";
    }
    if (context.hasHebrewWords) return "No listed CEFR words";
    return "Awaiting Hebrew words";
  }
  if (score >= 86) return "Advanced";
  if (score >= 66) return "Upper-intermediate";
  if (score >= 40) return "Intermediate";
  if (score > 0) return "Foundational";
  return "Awaiting JLPT words";
}

function getDefaultEasierTarget(distribution, language) {
  const config = languageConfigs[language];
  const dominantLevel = config.levels.reduce((currentBest, level) =>
    distribution[level] > distribution[currentBest] ? level : currentBest,
  );

  if (!distribution[dominantLevel]) return config.defaultTarget;
  const dominantIndex = config.levels.indexOf(dominantLevel);
  if (language === "hebrew") return config.levels[Math.max(0, dominantIndex - 1)];
  return config.levels[Math.min(config.levels.length - 1, dominantIndex + 1)];
}

function App() {
  const [language, setLanguage] = useState("japanese");
  const config = languageConfigs[language];
  const [text, setText] = useState("");
  const [messages, setMessages] = useState(getInitialMessages(language));
  const [chatInput, setChatInput] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  const [isAnnotating, setIsAnnotating] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [activeTextTab, setActiveTextTab] = useState("source");
  const [targetLevel, setTargetLevel] = useState(config.defaultTarget);
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
  const pendingTranslationsRef = useRef(new Set());

  const visibleEasierText = useMemo(
    () =>
      language === "hebrew"
        ? normalizeSourceForLanguage(easierText, "hebrew")
        : easierText,
    [easierText, language],
  );
  const visibleEasierAnnotations = easierAnnotations;
  const displayEasierText = useMemo(() => {
    if (language !== "japanese") return visibleEasierText;

    const japaneseInlinePattern = /([\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffー々〆〤。、！？「」『』（）［］【】・…])\s+([\u3040-\u30ff\u3400-\u9fff\uf900-\ufaffー々〆〤。、！？「」『』（）［］【】・…])/g;
    let normalized = visibleEasierText;
    while (true) {
      const compacted = normalized.replace(japaneseInlinePattern, "$1$2");
      if (compacted === normalized) break;
      normalized = compacted;
    }

    return normalized.trim();
  }, [language, visibleEasierText]);
  const hasVisibleEasierText = displayEasierText.trim() !== "";
  const easierRenderedTokens = getRenderedTokenData(
    visibleEasierAnnotations.length > 0
      ? visibleEasierAnnotations
      : displayEasierText
        ? [{ text: displayEasierText }]
        : [],
    { compactJapaneseWhitespace: true },
  );
  const hasCompletedEasierVersion =
    activeTextTab === "easier" &&
    !isSimplifying &&
    hasVisibleEasierText &&
    (language === "hebrew" || easierAnnotations.length > 0);
  const activeText = useMemo(
    () =>
      activeTextTab === "easier" && hasVisibleEasierText
        ? displayEasierText
        : text,
    [activeTextTab, displayEasierText, hasVisibleEasierText, text],
  );
  const activeAnnotations = useMemo(
    () =>
      language === "hebrew" && activeTextTab === "easier"
        ? isSimplifying
          ? []
          : visibleEasierAnnotations
        : hasCompletedEasierVersion
          ? visibleEasierAnnotations
          : annotations,
    [
      activeTextTab,
      annotations,
      hasCompletedEasierVersion,
      isSimplifying,
      language,
      visibleEasierAnnotations,
    ],
  );
  const sourceStats = useMemo(
    () => getTextStats(text, annotations, language),
    [text, annotations, language],
  );
  const stats = useMemo(
    () => getTextStats(activeText, activeAnnotations, language),
    [activeText, activeAnnotations, language],
  );
  const maxLevelCount = Math.max(1, ...Object.values(stats.distribution));
  const difficultyLabel = getDifficultyLabel(stats.difficultyScore, language, {
    hasGeneratedText: easierText.trim() !== "",
    hasHebrewWords: hebrewWordPattern.test(activeText),
    hasListedWords: stats.listedWordCount > 0,
    hasVisibleText: activeText.trim() !== "",
    isEasierTab: activeTextTab === "easier",
  });
  const defaultEasierTarget = useMemo(
    () => getDefaultEasierTarget(sourceStats.distribution, language),
    [sourceStats.distribution, language],
  );
  const sourcePattern = language === "hebrew" ? hebrewWordPattern : japaneseWordPattern;
  const sourceNeedsTokenization = sourcePattern.test(text) && annotations.length === 0;
  const isSourceTokenized =
    text.trim() !== "" && !isAnnotating && !sourceNeedsTokenization;
  function resetForLanguage(nextLanguage) {
    setLanguage(nextLanguage);
    setText("");
    setMessages(getInitialMessages(nextLanguage));
    setChatInput("");
    setUploadedFileName("");
    setActiveTextTab("source");
    setTargetLevel(languageConfigs[nextLanguage].defaultTarget);
    setEasierText("");
    setEasierTextSource("");
    setEasierAnnotations([]);
    setPendingSimplifyLevel(null);
    setAnnotations([]);
    setTranslations({});
    setActiveTooltip(null);
    setError("");
  }

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
          body: JSON.stringify({ text, language }),
          signal: controller.signal,
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail ?? `Could not annotate ${config.label} text.`);
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
  }, [text, language, config.label]);

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
        body: JSON.stringify({ word, language }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Translation unavailable");
      }
      setTranslations((currentTranslations) => ({
        ...currentTranslations,
        [word]: data.translation || "Translation unavailable",
      }));
      if (language === "japanese" && data.reading) {
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
    const level = language === "hebrew" ? token.cefr_level : token.jlpt_level ? `N${token.jlpt_level}` : null;
    setActiveTooltip({
      word: token.text,
      lookupWord,
      baseForm: token.base_form,
      level,
      gloss: token.english_gloss,
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

  function getRenderedTokenData(tokens, options = {}) {
    const renderedTokens = [];
    let previousHebrewSeparator = false;

    tokens.forEach((token, index) => {
      const level = language === "hebrew" ? token.cefr_level : token.jlpt_level ? `N${token.jlpt_level}` : null;
      const hasLevel = Boolean(level);
      const canLookup = canLookupToken(token, language);
      const isHebrewSeparator = language === "hebrew" && !canLookup;
      const tokenText =
        options.compactJapaneseWhitespace && language === "japanese"
          ? token.text.replace(/\s+/g, "")
          : token.text;

      if (!tokenText) return;
      if (isHebrewSeparator && previousHebrewSeparator) return;
      previousHebrewSeparator = isHebrewSeparator;
      if (!isHebrewSeparator) previousHebrewSeparator = false;

      renderedTokens.push({
        className: hasLevel
          ? `annotated-token ${config.classForLevel(level)}`
          : canLookup
            ? "lookup-token"
            : "plain-token",
        canLookup,
        index,
        text: isHebrewSeparator ? getHebrewSeparatorText(tokenText) : tokenText,
      });
    });

    return renderedTokens;
  }

  function renderAnnotatedText(editor, tokenList, fallbackText, options = {}) {
    if (!editor) return;

    editor.replaceChildren();
    const tokens = tokenList.length > 0 ? tokenList : fallbackText ? [{ text: fallbackText }] : [];

    getRenderedTokenData(tokens, options).forEach((token) => {
      const span = document.createElement("span");
      span.textContent = token.text;
      span.className = token.className;
      if (token.canLookup) {
        span.tabIndex = 0;
        span.dataset.tokenIndex = String(token.index);
      }
      editor.appendChild(span);
    });
  }

  useEffect(() => {
    if (activeTextTab !== "source") {
      editorRef.current?.replaceChildren();
      return;
    }
    renderAnnotatedText(editorRef.current, annotations, text);
  }, [activeTextTab, annotations, text, language]);

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
    const sourceText = normalizeSourceForLanguage(text, language);
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
        body: JSON.stringify({ text: sourceText, target_level: level, language }),
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
      let streamedText = "";

      const processStreamLine = async (line) => {
        if (!line.trim()) return;
        const event = JSON.parse(line);

        if (event.type === "chunk") {
          streamedText += event.text ?? "";
          setEasierText(normalizeSourceForLanguage(streamedText, language));
          await new Promise((resolve) => window.requestAnimationFrame(resolve));
        } else if (event.type === "done") {
          streamedText = event.text ?? streamedText;
          setEasierText(normalizeSourceForLanguage(streamedText, language));
          setEasierAnnotations(event.annotations ?? []);
          setEasierTextSource(sourceText);
        } else if (event.type === "error") {
          throw new Error(event.detail ?? "Could not create an easier version.");
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          await processStreamLine(line);
        }
      }

      buffer += decoder.decode();
      if (buffer.trim()) {
        await processStreamLine(buffer);
      }
    } catch (simplifyError) {
      setError(simplifyError.message);
    } finally {
      setIsSimplifying(false);
    }
  }

  function handleTextTabChange(tab) {
    if (language === "hebrew" && tab === "easier") {
      editorRef.current?.replaceChildren();
    }
    setActiveTextTab(tab);
    setActiveTooltip(null);
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
        body: JSON.stringify({ message, language }),
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
    setMessages(getInitialMessages(language));
    setError("");
  }

  return (
    <main className={`app-shell ${language === "hebrew" ? "hebrew-mode" : "japanese-mode"}`}>
      <nav className="top-nav" aria-label="App navigation">
        <div className="brand-mark">
          <span>{config.navLabel}</span>
          <strong>Readr</strong>
        </div>
        <div className="nav-actions">
          <div className="segmented-control" aria-label="Language mode">
            {Object.entries(languageConfigs).map(([key, item]) => (
              <button
                key={key}
                className={language === key ? "is-active" : ""}
                type="button"
                onClick={() => key !== language && resetForLanguage(key)}
                aria-pressed={language === key}
              >
                {item.shortLabel}
              </button>
            ))}
          </div>
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
          {config.info}
        </aside>
      )}

      {error && <div className="error-banner">{error}</div>}

      <section className="workspace">
        <aside className="panel chat-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Margin notes</p>
              <h2>{config.tutorTitle}</h2>
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
                {message.role === "assistant" ? (
                  <div className="message-content">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p>{message.content}</p>
                )}
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
              placeholder={config.chatPlaceholder}
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

        <section className="panel text-panel" data-active-tab={activeTextTab}>
          <div className="panel-header">
            <div>
              <p className="eyebrow">Study text</p>
              <h2>{activeTextTab === "source" ? config.textTitle : "AI easier version"}</h2>
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

          <div className="jlpt-legend" aria-label={`${config.scaleLabel} color legend`}>
            {config.levels.map((level) => (
              <span key={level} className={`legend-chip ${config.classForLevel(level)}`}>
                {level}
              </span>
            ))}
          </div>

          {activeTextTab === "source" ? (
            <div className="editor-frame" key="source-editor">
              <div
                ref={editorRef}
                className="study-editor"
                contentEditable
                data-placeholder={config.placeholder}
                dir={config.dir}
                lang={config.lang}
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
          ) : (
            <div className="easier-pane" key="easier-pane">
              <div className="easier-controls">
                <label htmlFor="target-level">Target difficulty</label>
                <select
                  id="target-level"
                  value={targetLevel}
                  onChange={handleTargetLevelChange}
                  disabled={isSimplifying || !text.trim()}
                >
                  {config.targetLevels.map((level) => (
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

              {(pendingSimplifyLevel || isSimplifying) && (
                <p className="easier-note">
                  {pendingSimplifyLevel
                    ? "Tokenizing the source text first…"
                    : `Starting a ${targetLevel} version…`}
                </p>
              )}
              {hasVisibleEasierText && isSimplifying && (
                <div className="easier-output empty-output streaming-output" dir={config.dir} lang={config.lang}>
                  <p>{displayEasierText}</p>
                </div>
              )}
              {hasVisibleEasierText && !isSimplifying && (
                <div className="editor-frame easier-output" dir={config.dir} lang={config.lang}>
                  <p
                    className="generated-reader"
                    dir={config.dir}
                    lang={config.lang}
                    onBlur={() => setActiveTooltip(null)}
                    onFocus={handleEditorPointer}
                    onMouseEnter={handleEditorPointer}
                    onMouseLeave={() => setActiveTooltip(null)}
                    onMouseMove={handleEditorPointer}
                    role="document"
                  >
                    {easierRenderedTokens.map((token) => {
                      const tokenProps = {
                        className: token.className,
                        key: `${token.index}-${token.text}`,
                      };
                      if (token.canLookup) {
                        tokenProps["data-token-index"] = token.index;
                        tokenProps.tabIndex = 0;
                      }
                      return <span {...tokenProps}>{token.text}</span>;
                    })}
                  </p>
                </div>
              )}
              {easierText && easierTextSource !== normalizeSourceForLanguage(text, language) && (
                <p className="easier-note">
                  Source text changed. Regenerate to refresh this version.
                </p>
              )}
            </div>
          )}
          {activeTooltip && (
            <div
              className={`token-popover is-visible ${activeTooltip.level ? config.classForLevel(activeTooltip.level) : "dictionary-token"}`}
              role="tooltip"
              style={{ left: activeTooltip.left, top: activeTooltip.top }}
            >
              <strong dir={config.dir}>
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
                    {activeTooltip.level}
                  </span>
                )}
              </strong>
              <span>
                {activeTooltip.gloss ?? translations[activeTooltip.lookupWord] ?? "Looking up…"}
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
              {language === "hebrew"
                ? "Vocabulary percentiles from the Hebrew CEFR list plus sentence-length signals"
                : "Vocabulary percentiles plus kanji density, sentence length, and grammar signals"}
            </small>
          </section>

          <section className="jlpt-chart" aria-label={`${config.scaleLabel} word distribution`}>
            <div className="chart-header">
              <span>{config.listedLabel}</span>
              <strong>
                {stats.leveledWordCount} / {stats.listedWordCount}
              </strong>
            </div>
            {config.chartLevels.map((level) => {
              const count = stats.distribution[level];
              const width = count
                ? `${Math.max(4, (count / maxLevelCount) * 100)}%`
                : "0%";
              return (
                <div className="chart-row" key={level}>
                  <span className={`chart-level ${config.classForLevel(level)}`}>
                    {level}
                  </span>
                  <div className="chart-track" aria-hidden="true">
                    <span
                      className={`chart-bar ${config.classForLevel(level)}`}
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
