// Initialize chat UI elements, API endpoints, and event listeners once the DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
    const messageForm = document.getElementById("message-form");
    const messageInput = document.getElementById("message-input");
    const chatWindow = document.getElementById("chat-window");
    const sendButton = document.getElementById("send-button");

    // Define backend API base URL, endpoint routes, and session chat history buffer
    const API_BASE_URL = "http://127.0.0.1:8000";
    const CHAT_URL = `${API_BASE_URL}/chat`;
    const FEEDBACK_URL = `${API_BASE_URL}/feedback`;
    const chatHistory = [];

    // Handle form submission to send user questions and display assistant responses
    messageForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const question = messageInput.value.trim();

        // Abort submission if the trimmed user input is empty
        if (question === "") return;

        // Render the user message, reset the input field, and show a typing indicator
        addMessageToChat(question, "user");
        messageInput.value = "";
        sendButton.disabled = true;
        const typingIndicator = addMessageToChat("O assistente está a pensar...", "bot", "typing-indicator");

        // Send the question and recent chat history to the backend chat endpoint
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

            // Throw an error if the HTTP response status indicates a server or network failure
            if (!response.ok) {
                throw new Error("Erro na rede ou no servidor.");
            }

            // Parse the JSON payload and remove the temporary typing indicator
            const data = await response.json();
            typingIndicator.remove();

            // Strip sources and medical disclaimers before saving the assistant turn to history
            const cleanAnswerForHistory = data.answer
                .split("\n\nFontes:")[0]
                .split("\n\nEsta informação não substitui")[0]
                .trim();

            // Append the user and assistant messages to session history and render the response
            chatHistory.push({ role: "user", content: question });
            chatHistory.push({ role: "assistant", content: cleanAnswerForHistory });
            addBotResponseWithFeedback(question, data.answer);

        // Remove typing indicator and display a fallback error message on request failure
        } catch (error) {
            typingIndicator.remove();
            addMessageToChat("Desculpe, algo correu mal. Tente novamente.", "bot");
            console.error("Erro ao contactar a API:", error);

        // Re-enable the send button and return focus to the message input field
        } finally {
            sendButton.disabled = false;
            messageInput.focus();
        }
    });

    // Escape HTML special characters and convert basic Markdown syntax into HTML tags
    function formatMarkdownToHtml(text) {
        let safeText = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Replace Markdown bold, italic, list bullets, and newlines with HTML equivalents
        safeText = safeText.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        safeText = safeText.replace(/\*(.+?)\*/g, "<em>$1</em>");
        safeText = safeText.replace(/^- (.+)$/gm, "• $1");
        safeText = safeText.replace(/\n/g, "<br>");
        return safeText;
    }

    // Create and append a plain text message bubble to the chat window
    function addMessageToChat(message, sender, id = null) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);

        // Apply the typing indicator style class when specified by the caller
        if (id === "typing-indicator") {
            messageElement.classList.add("typing-indicator");
        }

        // Set message text content, append to chat container, and scroll to bottom
        messageElement.textContent = message;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        return messageElement;
    }

    // Render a formatted bot response along with a negative feedback button
    function addBotResponseWithFeedback(question, rawAnswer) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", "bot-message");

        // Format the raw Markdown answer as HTML and attach it to the message container
        const contentDiv = document.createElement("div");
        contentDiv.innerHTML = formatMarkdownToHtml(rawAnswer);
        messageElement.appendChild(contentDiv);

        // Create the feedback container and the thumbs-down report button
        const actionsDiv = document.createElement("div");
        actionsDiv.classList.add("feedback-actions");
        const btnDown = document.createElement("button");
        btnDown.type = "button";
        btnDown.classList.add("feedback-btn");
        btnDown.title = "Reportar resposta incorreta";
        btnDown.textContent = "👎";

        // Create the status label and assemble the feedback action controls
        const statusSpan = document.createElement("span");
        statusSpan.classList.add("feedback-status");
        actionsDiv.appendChild(statusSpan);
        actionsDiv.appendChild(btnDown);
        messageElement.appendChild(actionsDiv);

        // Submit negative feedback to the backend API when the report button is clicked
        btnDown.addEventListener("click", async () => {
            btnDown.disabled = true;
            statusSpan.textContent = "Enviando...";

            // Post the question and raw answer payload to the feedback endpoint
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

                // Throw an error if the feedback endpoint returns a non-OK response
                if (!res.ok) throw new Error("Falha ao enviar feedback");
                statusSpan.textContent = "Reportado.";

            // Re-enable the button and display an error status if feedback submission fails
            } catch (err) {
                btnDown.disabled = false;
                statusSpan.textContent = "Erro ao enviar.";
                console.error("Erro no /feedback:", err);
            }
        });

        // Append the complete bot message element to the chat window and scroll down
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});