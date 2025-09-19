import React from 'react';

interface DiagramContainerProps {
  caption: string;
  children: React.ReactNode;
}

const DiagramContainer: React.FC<DiagramContainerProps> = ({ caption, children }) => {
  return (
    <figure className="my-8 not-prose bg-slate-50 dark:bg-slate-800/50 ring-1 ring-slate-200 dark:ring-slate-700 rounded-lg overflow-hidden">
      <div className="p-4 sm:p-6 flex justify-center items-center">
        {children}
      </div>
      <figcaption className="text-sm text-center text-slate-500 dark:text-slate-400 bg-white dark:bg-slate-900/50 px-4 py-3 border-t border-slate-200 dark:border-slate-700">
        {caption}
      </figcaption>
    </figure>
  );
};

export default DiagramContainer;
