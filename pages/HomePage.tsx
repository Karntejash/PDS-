import React from 'react';
import Card from '../components/Card';

const HomePage: React.FC = () => {
    const sections = [
        {
            title: 'Notes',
            description: 'In-depth notes covering every topic from the syllabus, with code examples.',
            linkTo: '/notes',
            icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>
        },
        {
            title: 'Flashcards',
            description: 'Reinforce your knowledge with interactive flashcards on key concepts and syntax.',
            linkTo: '/flashcards',
            icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
        },
        {
            title: 'Past Papers',
            description: 'Practice with detailed model answers for questions from previous years.',
            linkTo: '/pyp',
            icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg>
        },
        {
            title: 'Cheat Sheet',
            description: 'A quick reference guide with key terms, data structures, and library functions.',
            linkTo: '/cheatsheet',
            icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 4a2 2 0 114 0v1a1 1 0 001 1h3a1 1 0 011 1v3a1 1 0 01-1 1h-1a2 2 0 100 4h1a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1v-1a2 2 0 10-4 0v1a1 1 0 01-1 1H7a1 1 0 01-1-1v-3a1 1 0 011-1h1a2 2 0 100-4H7a1 1 0 01-1-1V8a1 1 0 011-1h3a1 1 0 001-1V4z" /></svg>
        },
        {
            title: 'AI Quiz Generator',
            description: 'Test your knowledge by generating custom quizzes for any unit with our Gemini-powered AI.',
            linkTo: '/quiz',
            icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
        }
    ];

  return (
    <div className="text-center">
      <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-slate-900 dark:text-white mb-6">Your AI-Powered Data Science Study Hub</h1>
      <p className="text-lg text-slate-600 dark:text-slate-300 max-w-3xl mx-auto mb-16 md:mb-20">
        Master Python for Data Science with detailed notes, interactive flashcards, past papers, and a powerful AI quiz generator. All the tools you need to succeed, in one place.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {sections.map(section => (
            <Card key={section.title} {...section} />
        ))}
      </div>
    </div>
  );
};

export default HomePage;