import React from 'react';

interface CodeBlockProps {
  code: string;
  language?: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = 'python' }) => {
  return (
    <div className="bg-slate-100 dark:bg-slate-800/50 rounded-lg my-6 overflow-hidden border border-slate-200 dark:border-slate-700 not-prose">
        <div className="flex justify-between items-center px-4 py-2 bg-slate-200/50 dark:bg-slate-700/50 text-xs text-slate-500 dark:text-slate-400">
            <span>{language.toUpperCase()}</span>
            <button
                onClick={() => navigator.clipboard.writeText(code.trim())}
                className="font-sans text-xs hover:text-slate-800 dark:hover:text-white transition-colors p-1 rounded hover:bg-slate-300/50 dark:hover:bg-slate-600/50"
            >
                Copy
            </button>
        </div>
        <pre className="p-4 text-sm overflow-x-auto text-slate-800 dark:text-slate-200">
            <code className={`language-${language}`}>
                {code.trim()}
            </code>
        </pre>
    </div>
  );
};

export default CodeBlock;