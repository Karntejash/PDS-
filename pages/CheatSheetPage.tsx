import React, { useState } from 'react';
import { cheatSheetData } from '../data/cheatSheetData';

const CheatSheetPage: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  const toggleSection = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div>
      <div className="text-center mb-12 md:mb-16">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white">Key Concepts & Cheat Sheet</h1>
        <p className="text-lg mt-4 text-slate-500 dark:text-slate-400">A quick reference guide for essential topics.</p>
      </div>

      <div className="space-y-4 max-w-4xl mx-auto">
        {cheatSheetData.map((section, index) => (
          <div key={index} className="bg-white dark:bg-slate-800/50 rounded-lg border border-slate-200 dark:border-slate-700/50 transition-all duration-300">
            <button
              onClick={() => toggleSection(index)}
              className="w-full flex justify-between items-center p-5 md:p-6 text-left"
            >
              <h2 className="text-lg md:text-xl font-bold text-sky-600 dark:text-sky-400">
                {section.title}
              </h2>
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className={`h-6 w-6 transform transition-transform duration-300 text-slate-500 ${openIndex === index ? 'rotate-180' : ''}`} 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <div className={`transition-all duration-500 ease-in-out overflow-hidden ${openIndex === index ? 'max-h-full' : 'max-h-0'}`}>
               <div className="p-5 md:p-6 border-t border-slate-200 dark:border-slate-700">
                <div className="prose dark:prose-invert max-w-none text-slate-700 dark:text-slate-300 text-justify">
                  {section.content}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CheatSheetPage;
