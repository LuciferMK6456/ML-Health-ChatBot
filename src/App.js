import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [userInput, setUserInput] = useState("");
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSend = async () => {
    if (userInput.trim() === "") {
      setError("Please enter a symptom.");
      return;
    }

    setConversation([...conversation, { sender: "user", text: userInput }]);
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post("http://localhost:5000/analyze", {
        symptom: userInput
      });

      setConversation([
        ...conversation,
        { sender: "user", text: userInput },
        { sender: "bot", text: response.data.message }
      ]);
    } catch (err) {
      setError("Failed to get a diagnosis. Please try again later.");
      setConversation([...conversation, { sender: "bot", text: "Error: Could not process request" }]);
    }

    setLoading(false);
    setUserInput("");
  };

  return (
    <div className="app">
      <h1>Health Chatbot</h1>
      <div className="chat-window">
        {conversation.map((msg, index) => (
          <div key={index} className={msg.sender === "user" ? "user-message" : "bot-message"}>
            {msg.text}
          </div>
        ))}
        {loading && <div className="bot-message">Loading...</div>}
      </div>
      {error && <div className="error-message">{error}</div>}
      <div className="input-section">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder="Enter your symptoms..."
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}

export default App;
