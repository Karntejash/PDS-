import React, { useState, useCallback } from 'react';
import { syllabus } from '../data/syllabus';
import { pypData } from '../data/pypData';
import { generateQuiz, generateQuizFromPaper } from '../services/geminiService';
import { QuizData, Unit, PYPaper } from '../types';
import Quiz from '../components/Quiz';

// Type guard to check if an object is a Unit
function isUnit(source: Unit | PYPaper): source is Unit {
    return (source as Unit).topics !== undefined;
}

const QuizGeneratorPage: React.FC = () => {
  const [selectedSourceId, setSelectedSourceId] = useState<string>('unit-1'); // Default to first unit
  const [quizData, setQuizData] = useState<QuizData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const getSelectedSource = useCallback((): Unit | PYPaper | null => {
    const [type, id] = selectedSourceId.split('-');
    if (type === 'unit') {
      return syllabus.find(u => u.id === parseInt(id, 10)) || null;
    }
    if (type === 'pyp') {
      return pypData.find(p => p.year === id) || null;
    }
    return null;
  }, [selectedSourceId]);

  const handleGenerateQuiz = useCallback(async () => {
    const selectedSource = getSelectedSource();
    if (!selectedSource) return;

    setIsLoading(true);
    setError('');
    setQuizData(null);
    try {
      const result = isUnit(selectedSource)
        ? await generateQuiz(selectedSource)
        : await generateQuizFromPaper(selectedSource);
      setQuizData(result);
    } catch (err) {
        if (err instanceof Error) {
            setError(err.message);
        } else {
            setError('An unknown error occurred.');
        }
    } finally {
      setIsLoading(false);
    }
  }, [getSelectedSource]);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12 md:mb-16">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white">AI Quiz Generator</h1>
        <p className="text-lg mt-4 text-slate-500 dark:text-slate-400">Test your knowledge with custom quizzes powered by Gemini.</p>
      </div>
      
      {!quizData ? (
        <div className="bg-white dark:bg-slate-800/50 p-8 md:p-12 rounded-lg shadow-sm text-center border border-slate-200 dark:border-slate-800">
          <h2 className="text-2xl font-semibold mb-4 text-slate-900 dark:text-white">Test Your Knowledge</h2>
          <p className="text-slate-600 dark:text-slate-300 mb-8 max-w-2xl mx-auto">Select a unit or a past paper and let our AI create a custom 10-question quiz for you. It's a great way to check your understanding and prepare for exams.</p>
          
          <div className="mb-8">
            <label htmlFor="source-select" className="block text-sm font-medium text-slate-700 dark:text-slate-200 mb-2">
              Choose a Quiz Source:
            </label>
            <select
              id="source-select"
              value={selectedSourceId}
              onChange={(e) => setSelectedSourceId(e.target.value)}
              className="w-full max-w-md mx-auto p-3 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 focus:ring-sky-500 focus:border-sky-500"
            >
              <optgroup label="Syllabus Units">
                {syllabus.map(unit => (
                  <option key={`unit-${unit.id}`} value={`unit-${unit.id}`}>{unit.title}</option>
                ))}
              </optgroup>
              <optgroup label="Past Papers">
                {pypData.map(paper => (
                  <option key={`pyp-${paper.year}`} value={`pyp-${paper.year}`}>{`Past Paper - ${paper.year}`}</option>
                ))}
              </optgroup>
            </select>
          </div>

          <button
            onClick={handleGenerateQuiz}
            disabled={isLoading || !selectedSourceId}
            className="bg-sky-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-sky-700 transition-colors disabled:bg-sky-400 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Generating Quiz...' : 'Generate Quiz'}
          </button>
          
          {isLoading && <p className="mt-4 text-slate-500">Please wait, the AI is crafting your questions...</p>}
          {error && <p className="mt-4 text-red-500 font-semibold">{error}</p>}
        </div>
      ) : (
        <Quiz quizData={quizData} />
      )}
    </div>
  );
};

export default QuizGeneratorPage;