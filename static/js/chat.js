/* ==========================================================================
   CrimeVision AI - Grounded Forensic Chat Interface
   ========================================================================== */

window.CV = window.CV || {};

CV.chat = (function() {
  const chatMessagesEl = document.getElementById('chatMessages');
  const chatFormEl = document.getElementById('chatForm');
  const chatInputEl = document.getElementById('chatInput');
  const chatTypingEl = document.getElementById('chatTyping');
  const chatChipsEl = document.getElementById('chatChips');

  /**
   * Escape and safely create a chat bubble element.
   */
  function appendMessage(role, text, sources) {
    if (!chatMessagesEl) return;

    const msgContainer = document.createElement('div');
    msgContainer.className = `cv-msg ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'cv-msg-bubble';
    bubble.textContent = text;
    msgContainer.appendChild(bubble);

    if (Array.isArray(sources) && sources.length > 0) {
      const sourcesWrap = document.createElement('div');
      sourcesWrap.className = 'cv-msg-sources';

      sources.forEach(function(src) {
        const tag = document.createElement('span');
        tag.className = 'cv-source-tag';
        tag.textContent = src;

        // If source is a timestamp, allow clicking to seek
        const timeMatch = String(src).match(/(\d+(?:\.\d+)?)\s*s/);
        if (timeMatch && typeof CV.seekMedia === 'function') {
          tag.style.cursor = 'pointer';
          tag.title = `Seek media to ${timeMatch[1]}s`;
          tag.addEventListener('click', function() {
            CV.seekMedia(parseFloat(timeMatch[1]));
          });
        }

        sourcesWrap.appendChild(tag);
      });

      msgContainer.appendChild(sourcesWrap);
    }

    chatMessagesEl.appendChild(msgContainer);
    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  }

  function setTyping(isTyping) {
    if (!chatTypingEl) return;
    if (isTyping) {
      chatTypingEl.classList.add('active');
    } else {
      chatTypingEl.classList.remove('active');
    }
    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  }

  /**
   * Send a query message to /api/chat.
   */
  async function sendMessage(text) {
    const cleanText = (text || '').trim();
    if (!cleanText) return;

    if (!CV.state || !CV.state.currentSessionId) {
      if (CV.ui && typeof CV.ui.showToast === 'function') {
        CV.ui.showToast('Please ingest or load an evidence session first.', 'warning');
      }
      return;
    }

    // 1. Render user message
    appendMessage('user', cleanText);
    CV.state.chatHistory = CV.state.chatHistory || [];
    CV.state.chatHistory.push({ role: 'user', content: cleanText });

    if (chatInputEl) {
      chatInputEl.value = '';
    }

    setTyping(true);

    try {
      const payload = {
        session_id: CV.state.currentSessionId,
        message: cleanText,
        history: CV.state.chatHistory.slice(-6)
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server responded with ${response.status}`);
      }

      const answer = data.answer || 'No response generated.';
      const sources = data.sources || [];

      appendMessage('assistant', answer, sources);
      CV.state.chatHistory.push({ role: 'assistant', content: answer });

    } catch (err) {
      console.error('Chat error:', err);
      appendMessage('assistant', `Error processing request: ${err.message}`);
      if (CV.ui && typeof CV.ui.showToast === 'function') {
        CV.ui.showToast(`Chat error: ${err.message}`, 'error');
      }
    } finally {
      setTyping(false);
    }
  }

  /**
   * Initialize chat event listeners.
   */
  function initChat() {
    if (chatFormEl) {
      chatFormEl.addEventListener('submit', function(e) {
        e.preventDefault();
        if (chatInputEl) {
          sendMessage(chatInputEl.value);
        }
      });
    }

    if (chatChipsEl) {
      chatChipsEl.addEventListener('click', function(e) {
        const chip = e.target.closest('.cv-chip');
        if (chip && chip.dataset.query) {
          if (chatInputEl) {
            chatInputEl.value = chip.dataset.query;
          }
          sendMessage(chip.dataset.query);
        }
      });
    }
  }

  return {
    initChat: initChat,
    sendMessage: sendMessage,
    appendMessage: appendMessage
  };
})();
