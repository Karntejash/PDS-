import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 mt-auto">
      <div className="container mx-auto px-4 py-6 text-center">
        <p>&copy; {new Date().getFullYear()} Python for Data Science Hub. All rights reserved.</p>
        <p className="text-sm mt-1">An AI-powered learning assistant.</p>
      </div>
    </footer>
  );
};

export default Footer;