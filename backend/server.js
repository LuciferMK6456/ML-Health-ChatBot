const express = require("express");
const cors = require("cors");
const { PythonShell } = require("python-shell");

const app = express();
app.use(cors());
app.use(express.json());

app.post("/analyze", (req, res) => {
    const symptom = req.body.symptom;

    if (!symptom || symptom.trim() === "") {
        return res.status(400).json({ message: "Symptom input is required" });
    }

    let options = {
        args: [symptom],
    };

    PythonShell.run("ml_model.py", options, function (err, results) {
        if (err) {
            console.error("Error in Python script:", err);
            return res.status(500).json({ message: "Internal server error" });
        }

        try {
            const response = JSON.parse(results[0]);
            if (response.error) {
                return res.status(400).json({ message: response.error });
            }

            res.json({
                message: `Diagnosis: ${response.disease}, Treatment: ${response.treatment}, Diet: ${response.diet}`,
            });
        } catch (parseError) {
            console.error("Error parsing Python script output:", parseError);
            return res.status(500).json({ message: "Error processing the diagnosis" });
        }
    });
});

app.use((req, res) => {
    res.status(404).json({ message: "Endpoint not found" });
});

app.listen(5000, () => {
    console.log("Server is running on http://localhost:5000");
});
