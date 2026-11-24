document.addEventListener("DOMContentLoaded", () => {
    const messageForm = document.getElementById("message-form");
    const messageInput = document.getElementById("message-input");
    const chatWindow = document.getElementById("chat-window");
    const sendButton = document.getElementById("send-button");

    const API_BASE_URL = "http://127.0.0.1:8000";
    const CHAT_URL = `${API_BASE_URL}/chat`;
    const FEEDBACK_URL = `${API_BASE_URL}/feedback`;

    // Janela curta de memória da sessão atual (limpa ao recarregar a página)
    const chatHistory = [];

    messageForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = messageInput.value.trim();
        if (question === "") return;

        addMessageToChat(question, "user");
        messageInput.value = "";
        sendButton.disabled = true;

        const typingIndicator = addMessageToChat("O assistente está a pensar...", "bot", "typing-indicator");

        try {
            const response = await fetch(CHAT_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    question,
                    history: chatHistory.slice(-4)
                })
            });

            if (!response.ok) {
                throw new Error("Erro na rede ou no servidor.");
            }

            const data = await response.json();
            typingIndicator.remove();

            // Guarda o turno atual (removendo rodapé de Fontes/Aviso para manter o histórico limpo)
            const cleanAnswerForHistory = data.answer
                .split("\n\nFontes:")[0]
                .split("\n\nEsta informação não substitui")[0]
                .trim();

            chatHistory.push({ role: "user", content: question });
            chatHistory.push({ role: "assistant", content: cleanAnswerForHistory });

            addBotResponseWithFeedback(question, data.answer);
        } catch (error) {
            typingIndicator.remove();
            addMessageToChat("Desculpe, algo correu mal. Tente novamente.", "bot");
            console.error("Erro ao contactar a API:", error);
        } finally {
            sendButton.disabled = false;
            messageInput.focus();
        }
    });

    function formatMarkdownToHtml(text) {
        // Escapa HTML básico para evitar XSS
        let safeText = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Converte negrito (**texto**) e itálico (*texto*)
        safeText = safeText.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        safeText = safeText.replace(/\*(.+?)\*/g, "<em>$1</em>");

        // Converte linhas de lista (- item) e quebras de linha
        safeText = safeText.replace(/^- (.+)$/gm, "• $1");
        safeText = safeText.replace(/\n/g, "<br>");

        return safeText;
    }

    function addMessageToChat(message, sender, id = null) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);

        if (id === "typing-indicator") {
            messageElement.classList.add("typing-indicator");
        }

        messageElement.textContent = message;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        return messageElement;
    }

    function addBotResponseWithFeedback(question, rawAnswer) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", "bot-message");

        const contentDiv = document.createElement("div");
        contentDiv.innerHTML = formatMarkdownToHtml(rawAnswer);
        messageElement.appendChild(contentDiv);

        const actionsDiv = document.createElement("div");
        actionsDiv.classList.add("feedback-actions");

        const btnDown = document.createElement("button");
        btnDown.type = "button";
        btnDown.classList.add("feedback-btn");
        btnDown.title = "Reportar resposta incorreta";
        btnDown.textContent = "👎";

        const statusSpan = document.createElement("span");
        statusSpan.classList.add("feedback-status");

        actionsDiv.appendChild(statusSpan);
        actionsDiv.appendChild(btnDown);
        messageElement.appendChild(actionsDiv);

        btnDown.addEventListener("click", async () => {
            btnDown.disabled = true;
            statusSpan.textContent = "Enviando...";

            try {
                const res = await fetch(FEEDBACK_URL, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        question,
                        answer: rawAnswer
                    })
                });

                if (!res.ok) throw new Error("Falha ao enviar feedback");
                statusSpan.textContent = "Reportado.";
            } catch (err) {
                btnDown.disabled = false;
                statusSpan.textContent = "Erro ao enviar.";
                console.error("Erro no /feedback:", err);
            }
        });

        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});