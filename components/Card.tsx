import React from 'react';
import { Link } from 'react-router-dom';

interface CardProps {
  title: string;
  description: string;
  linkTo: string;
  icon: React.ReactNode;
}

const Card: React.FC<CardProps> = ({ title, description, linkTo, icon }) => {
  return (
    <Link to={linkTo} className="block group">
      <div className="bg-white dark:bg-slate-800/50 rounded-lg p-8 h-full flex flex-col items-start transition-all duration-300 hover:shadow-lg hover:shadow-sky-100 dark:hover:shadow-sky-900/20 border border-slate-200 dark:border-slate-800 hover:border-sky-300 dark:hover:border-sky-700">
        <div className="bg-sky-100 text-sky-600 dark:bg-sky-900/50 dark:text-sky-400 rounded-lg p-3 mb-5">
          {icon}
        </div>
        <h3 className="text-xl font-bold mb-3 text-slate-900 dark:text-white">{title}</h3>
        <p className="text-slate-600 dark:text-slate-300 flex-grow">{description}</p>
        <span className="mt-6 text-sky-600 dark:text-sky-400 group-hover:text-sky-500 dark:group-hover:text-sky-300 font-semibold text-sm">
          Go to section &rarr;
        </span>
      </div>
    </Link>
  );
};

export default Card;