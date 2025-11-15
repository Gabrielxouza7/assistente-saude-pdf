// script.js
document.addEventListener("DOMContentLoaded", () => {
    
    const messageForm = document.getElementById("message-form");
    const messageInput = document.getElementById("message-input");
    const chatWindow = document.getElementById("chat-window");
    const sendButton = document.getElementById("send-button");
    
    const API_URL = "http://127.0.0.1:8000/chat";

    messageForm.addEventListener("submit", async (e) => {
        e.preventDefault(); 
        const question = messageInput.value.trim();
        if (question === "") return; 

        addMessageToChat(question, "user");
        messageInput.value = "";
        const typingIndicator = addMessageToChat("O assistente está a pensar...", "bot", "typing-indicator");

        try {
            // 2. Envia a pergunta
            const response = await fetch(API_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    "question": question
                })
            });

            if (!response.ok) {
                throw new Error("Erro na rede ou no servidor.");
            }

            const data = await response.json();
            typingIndicator.remove();

            // 4. Mostra a resposta do bot
            const botAnswer = data.answer.replace(/\n/g, "<br>");
            addMessageToChat(botAnswer, "bot");

        } catch (error) {
            typingIndicator.remove();
            addMessageToChat("Desculpe, algo correu mal. Tente novamente.", "bot");
            console.error("Erro ao contactar a API:", error);
        }
    });

    /**
     * Função auxiliar para adicionar mensagens à janela de chat
     */
    function addMessageToChat(message, sender, id = null) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        
        if (id === "typing-indicator") {
            messageElement.classList.add("typing-indicator");
        }
        
        messageElement.innerHTML = message;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
        
        return messageElement;
    }
});