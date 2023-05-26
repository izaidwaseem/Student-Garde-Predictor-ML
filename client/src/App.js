import React, { useState } from "react";
import axios from 'axios';

function App() {
  const [mid1, setMid1] = useState("");
  const [mid2, setMid2] = useState("");
  const [quiz, setQuiz] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async event => {
    event.preventDefault();

    const payload = {
      mid1: Number(mid1),
      mid2: Number(mid2),
      quiz: Number(quiz)
    };
    
    // Make a POST request to the Flask server
    const res = await axios.post('http://127.0.0.1:8080/predict', payload);

    // Set the result state with the received prediction
    setResult(res.data);
  };

  return (
    <div className="container d-flex justify-content-center align-items-center vh-100">
      <div className="card p-4">
        <h1 className="mb-4 text-center">Student Grade Predictor</h1>
        <form onSubmit={handleSubmit}>
          <div className="mb-3 text-center">
            <label htmlFor="mid1" className="form-label">Mid1:</label>
            <input type="number" className="form-control" id="mid1" value={mid1} onChange={e => setMid1(e.target.value)} />
          </div>
          <div className="mb-3 text-center">
            <label htmlFor="mid2" className="form-label">Mid2:</label>
            <input type="number" className="form-control" id="mid2" value={mid2} onChange={e => setMid2(e.target.value)} />
          </div>
          <div className="mb-3 text-center">
            <label htmlFor="quiz" className="form-label">Quizzes:</label>
            <input type="number" className="form-control" id="quiz" value={quiz} onChange={e => setQuiz(e.target.value)} />
          </div>
          <button type="submit" className="btn btn-primary">Predict</button>
        </form>
        {result && (
          <div className="mt-4">
            <p>Final Score: {result.final_score}</p>
            <p>Final Grade: {result.final_grade}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
