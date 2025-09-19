import React, { useState, useCallback } from 'react';
import { pypData } from '../data/pypData';
import CodeBlock from '../components/CodeBlock';
import { predictExamQuestions, generateQuestionSummary } from '../services/geminiService';
import { syllabus } from '../data/syllabus';
import { PYPQuestion } from '../types';
import { parseMarkdownToHTML } from '../App';
import DiagramContainer from '../components/DiagramContainer';


const PYPPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(pypData[0].year);
  const [predictions, setPredictions] = useState('');
  const [isLoadingPredictions, setIsLoadingPredictions] = useState(false);
  const [error, setError] = useState('');
  const [summaries, setSummaries] = useState<Record<string, { summary?: string; isLoading: boolean; error?: string }>>({});

  const handlePredictQuestions = useCallback(async () => {
    setIsLoadingPredictions(true);
    setError('');
    setPredictions('');
    try {
      const result = await predictExamQuestions(syllabus, pypData);
      setPredictions(result);
    } catch (err) {
      setError('Failed to predict questions. Please try again.');
    } finally {
      setIsLoadingPredictions(false);
    }
  }, []);
  
  const handleGenerateSummary = useCallback(async (question: PYPQuestion) => {
    setSummaries(prev => ({ ...prev, [question.id]: { isLoading: true } }));
    try {
        const result = await generateQuestionSummary(question);
        setSummaries(prev => ({ ...prev, [question.id]: { isLoading: false, summary: result } }));
    } catch (err) {
        setSummaries(prev => ({ ...prev, [question.id]: { isLoading: false, error: 'Failed to generate summary.' } }));
    }
  }, []);


  return (
    <div className="max-w-5xl mx-auto">
      <div className="text-center mb-12 md:mb-16">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white">Past Papers & Predictions</h1>
        <p className="text-lg mt-4 text-slate-500 dark:text-slate-400 max-w-3xl mx-auto">Practice with model answers from previous years and get AI-powered exam predictions.</p>
      </div>

      <div className="bg-white dark:bg-slate-800/50 p-6 md:p-8 rounded-lg shadow-sm mb-12 border border-slate-200 dark:border-slate-800">
        <h2 className="text-2xl font-bold mb-4 text-sky-600 dark:text-sky-400">AI Exam Question Predictor</h2>
        <p className="text-slate-600 dark:text-slate-300 mb-6 max-w-3xl">Let Gemini analyze the syllabus and past papers to predict high-probability questions for your next exam.</p>
        <button
          onClick={handlePredictQuestions}
          disabled={isLoadingPredictions}
          className="bg-sky-600 text-white px-6 py-3 rounded-lg hover:bg-sky-700 transition-colors disabled:bg-sky-400 disabled:cursor-not-allowed font-semibold"
        >
          {isLoadingPredictions ? 'Predicting...' : 'Predict Exam Questions'}
        </button>
        {isLoadingPredictions && <p className="mt-4 text-slate-500">Analyzing patterns and generating questions, please wait...</p>}
        {error && <p className="mt-4 text-red-500">{error}</p>}
        {predictions && (
          <div className="mt-6 p-6 bg-sky-50 dark:bg-sky-900/20 rounded-md border border-sky-200 dark:border-sky-900">
             <h3 className="text-lg font-semibold mb-4 text-slate-900 dark:text-white">Predicted Questions:</h3>
             <div className="prose dark:prose-invert max-w-none text-justify" dangerouslySetInnerHTML={{ __html: parseMarkdownToHTML(predictions) }} />
          </div>
        )}
      </div>

      <div className="mb-8 flex justify-center border-b border-slate-200 dark:border-slate-700">
        {pypData.map((paper) => (
          <button
            key={paper.year}
            onClick={() => setActiveTab(paper.year)}
            className={`px-6 py-3 text-lg font-medium transition-colors -mb-px ${
              activeTab === paper.year
                ? 'border-b-2 border-sky-500 text-sky-600 dark:text-sky-400'
                : 'text-slate-500 hover:text-sky-600 dark:hover:text-sky-400 border-b-2 border-transparent'
            }`}
          >
            {`Test ${paper.year}`}
          </button>
        ))}
      </div>
      
      <div>
        {pypData.map((paper) => (
          <div key={paper.year} className={activeTab === paper.year ? 'block' : 'hidden'}>
            <div className="space-y-10">
              {paper.questions.map((q) => (
                <div key={q.id} className="bg-white dark:bg-slate-800/50 p-6 md:p-8 rounded-lg shadow-sm border border-slate-200 dark:border-slate-800">
                  <h3 className="text-xl font-semibold mb-6 text-slate-800 dark:text-gray-100">{q.question}</h3>
                  <div className="prose prose-lg dark:prose-invert max-w-none text-slate-700 dark:text-slate-300 text-justify">
                    {q.answer.diagram && (
                      <DiagramContainer caption={q.answer.diagram.caption}>
                        {q.answer.diagram.component}
                      </DiagramContainer>
                    )}
                    {q.answer.theory && <div dangerouslySetInnerHTML={{ __html: parseMarkdownToHTML(q.answer.theory) }} />}
                    {q.answer.code && <CodeBlock code={q.answer.code} />}
                    {q.answer.output && (
                      <div className="not-prose mt-4">
                        <h4 className="font-semibold text-sm text-slate-500 dark:text-slate-400">OUTPUT:</h4>
                        <pre className="bg-slate-100 dark:bg-slate-800/50 p-4 rounded-md text-sm text-slate-600 dark:text-slate-300 overflow-x-auto border border-slate-200 dark:border-slate-700 mt-2"><code>{q.answer.output}</code></pre>
                      </div>
                    )}
                  </div>

                   <div className="mt-6 border-t border-slate-200 dark:border-slate-700 pt-6">
                      <button
                          onClick={() => handleGenerateSummary(q)}
                          disabled={summaries[q.id]?.isLoading}
                          className="bg-sky-100 text-sky-700 dark:bg-sky-900/50 dark:text-sky-300 px-4 py-2 rounded-lg hover:bg-sky-200 dark:hover:bg-sky-800/60 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-sm font-semibold"
                      >
                          {summaries[q.id]?.isLoading ? 'Generating...' : 'Generate AI Summary'}
                      </button>
                      {summaries[q.id]?.isLoading && <p className="mt-4 text-sm text-slate-500">Generating summary...</p>}
                      {summaries[q.id]?.error && <p className="mt-4 text-sm text-red-500">{summaries[q.id]?.error}</p>}
                      {summaries[q.id]?.summary && (
                          <div className="mt-4 p-4 bg-sky-50 dark:bg-sky-900/20 rounded-md border border-sky-200 dark:border-sky-900">
                              <h4 className="text-base font-semibold mb-2 text-slate-900 dark:text-white">AI Summary:</h4>
                              <div className="prose dark:prose-invert max-w-none text-sm text-justify"
                                  dangerouslySetInnerHTML={{ __html: parseMarkdownToHTML(summaries[q.id]!.summary!) }} />
                          </div>
                      )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PYPPage;