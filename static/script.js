/* ============================================================
   SIGNLINGUA — Frontend Logic
   ============================================================ */

// Send a command to the Flask backend
async function sendCommand(action, extraData = {}) {
    try {
        await fetch('/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, ...extraData })
        });
    } catch (e) {
        console.error('Command error:', e);
    }
}

// Change language in the backend
async function changeLanguage(lang) {
    await sendCommand('language', { language: lang });
}
