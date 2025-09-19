import React, { useState, useCallback } from 'react';
import { syllabus } from '../data/syllabus';
import CodeBlock from '../components/CodeBlock';
import { generateUnitSummary } from '../services/geminiService';
import { Unit } from '../types';
import { parseMarkdownToHTML } from '../App';
import DiagramContainer from '../components/DiagramContainer';


const NotesPage: React.FC = () => {
  const [selectedUnit, setSelectedUnit] = useState<Unit | null>(syllabus[0]);
  const [summary, setSummary] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleGenerateSummary = useCallback(async () => {
    if (!selectedUnit) return;
    setIsLoading(true);
    setError('');
    setSummary('');
    try {
      const result = await generateUnitSummary(selectedUnit);
      setSummary(result);
    } catch (err) {
      setError('Failed to generate summary. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedUnit]);
  
  const handleUnitSelect = (unitId: number) => {
    const unit = syllabus.find(u => u.id === unitId) || null;
    setSelectedUnit(unit);
    setSummary('');
    setError('');
  };

  return (
    <div>
      <div className="text-center mb-12 md:mb-16">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white">Course Notes</h1>
        <p className="text-lg mt-4 text-slate-500 dark:text-slate-400">In-depth notes for every unit in the syllabus.</p>
      </div>

      {/* Tab Navigation */}
       <div className="mb-8 flex justify-center border-b border-slate-200 dark:border-slate-700">
        {syllabus.map((unit) => (
          <button
            key={unit.id}
            onClick={() => handleUnitSelect(unit.id)}
            className={`px-4 sm:px-6 py-3 text-base font-medium transition-colors -mb-px ${
              selectedUnit?.id === unit.id
                ? 'border-b-2 border-sky-500 text-sky-600 dark:text-sky-400'
                : 'text-slate-500 hover:text-sky-600 dark:hover:text-sky-400 border-b-2 border-transparent'
            }`}
          >
            {`Unit ${unit.id}`}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <main>
        {selectedUnit ? (
          <div className="bg-white dark:bg-slate-800/50 p-6 md:p-10 rounded-lg border border-slate-200 dark:border-slate-800 max-w-4xl mx-auto">
            <div className="flex flex-col sm:flex-row justify-between sm:items-center mb-8 border-b border-slate-200 dark:border-slate-700 pb-8">
              <h2 className="text-3xl font-bold text-sky-600 dark:text-sky-400 mb-4 sm:mb-0 max-w-md">{selectedUnit.title}</h2>
              <button
                onClick={handleGenerateSummary}
                disabled={isLoading}
                className="bg-sky-600 text-white px-5 py-2.5 rounded-lg hover:bg-sky-700 disabled:bg-sky-400 disabled:cursor-not-allowed transition-colors flex-shrink-0 font-semibold"
              >
                {isLoading ? 'Generating...' : 'Generate AI Summary'}
              </button>
            </div>

            {isLoading && <p className="text-center p-4">Generating summary, please wait...</p>}
            {error && <p className="text-center p-4 text-red-500">{error}</p>}
            {summary && (
              <div className="prose prose-lg dark:prose-invert max-w-none bg-sky-50 dark:bg-sky-900/20 p-6 rounded-lg mb-10 border border-sky-200 dark:border-sky-900 text-justify">
                <h3 className="text-xl font-semibold !mt-0">AI Generated Summary</h3>
                <div dangerouslySetInnerHTML={{ __html: parseMarkdownToHTML(summary) }} />
              </div>
            )}

            <div className="space-y-12">
              {selectedUnit.topics.map((topic, index) => (
                <div key={index} className="border-b border-slate-200 dark:border-slate-700 pb-8 last:border-b-0 last:pb-0">
                  <h3 className="text-2xl font-semibold mb-4 text-slate-800 dark:text-white">{topic.title}</h3>
                  <div 
                    className="prose prose-lg dark:prose-invert max-w-none text-slate-700 dark:text-slate-300 text-justify"
                    dangerouslySetInnerHTML={{ __html: parseMarkdownToHTML(topic.content) }}
                  />
                  {topic.diagram && (
                    <DiagramContainer caption={topic.diagram.caption}>
                      {topic.diagram.component}
                    </DiagramContainer>
                  )}
                  {topic.code && <CodeBlock code={topic.code} />}
                  {topic.output && (
                      <div className="mt-4">
                          <h4 className="font-semibold text-sm text-slate-500 dark:text-slate-400">OUTPUT:</h4>
                          <pre className="bg-slate-100 dark:bg-slate-800/50 p-4 rounded-md text-sm text-slate-600 dark:text-slate-300 overflow-x-auto border border-slate-200 dark:border-slate-700 mt-2">
                              <code>{topic.output}</code>
                          </pre>
                      </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-center text-lg text-slate-500">Please select a unit to view its notes.</p>
        )}
      </main>
    </div>
  );
};

export default NotesPage;