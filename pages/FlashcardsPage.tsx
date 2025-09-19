import React, { useState, useMemo } from 'react';
import { flashcardDecks } from '../data/flashcardData';
import Flashcard from '../components/Flashcard';
import { FlashcardDeck } from '../types';

const FlashcardsPage: React.FC = () => {
  const [selectedDeck, setSelectedDeck] = useState<FlashcardDeck | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);

  const currentCards = useMemo(() => selectedDeck?.cards || [], [selectedDeck]);

  const handleSelectDeck = (deck: FlashcardDeck) => {
    setSelectedDeck(deck);
    setCurrentIndex(0);
    setIsFlipped(false);
  };
  
  const handleReset = () => {
      setSelectedDeck(null);
      setCurrentIndex(0);
      setIsFlipped(false);
  }

  const handleNext = () => {
    if (currentIndex < currentCards.length - 1) {
        setIsFlipped(false);
        setTimeout(() => {
            setCurrentIndex((prevIndex) => prevIndex + 1);
        }, 150); 
    }
  };

  const handlePrev = () => {
    if (currentIndex > 0) {
        setIsFlipped(false);
        setTimeout(() => {
            setCurrentIndex((prevIndex) => prevIndex - 1);
        }, 150);
    }
  };
  
  if (!selectedDeck) {
    return (
        <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-white mb-4">Last-Minute Revision Flashcards</h1>
            <p className="text-lg text-slate-500 dark:text-slate-400 mb-12">Select a deck to review high-yield, exam-focused topics.</p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {flashcardDecks.map(deck => (
                    <button key={deck.id} onClick={() => handleSelectDeck(deck)} className="block group text-left">
                        <div className="bg-white dark:bg-slate-800/50 rounded-lg p-6 h-full flex flex-col items-start transition-all duration-300 hover:shadow-lg hover:shadow-sky-100 dark:hover:shadow-sky-900/20 border border-slate-200 dark:border-slate-800 hover:border-sky-300 dark:hover:border-sky-700">
                           <h3 className="text-lg font-bold mb-2 text-sky-600 dark:text-sky-400">{deck.title}</h3>
                           <p className="text-slate-600 dark:text-slate-300 flex-grow text-sm">{deck.description}</p>
                           <span className="mt-4 text-slate-500 dark:text-slate-400 group-hover:text-sky-600 dark:group-hover:text-sky-400 font-semibold text-sm">
                            Start Review &rarr;
                           </span>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    )
  }

  const currentCard = currentCards[currentIndex];

  return (
    <div className="max-w-2xl mx-auto text-center">
        <button onClick={handleReset} className="mb-6 text-sm font-semibold text-sky-600 dark:text-sky-400 hover:underline">
         &larr; Back to Decks
        </button>
      <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white mb-4">{selectedDeck.title}</h1>
      <p className="text-base text-slate-500 dark:text-slate-400 mb-8">Click on a card to flip it. Use the buttons to navigate.</p>
      
      <Flashcard
        question={currentCard.question}
        answer={currentCard.answer}
        isFlipped={isFlipped}
        onFlip={() => setIsFlipped(!isFlipped)}
      />

      <div className="mt-8">
        <p className="font-medium text-slate-500 dark:text-slate-400">
          Card {currentIndex + 1} of {currentCards.length}
        </p>
      </div>

      <div className="flex justify-center items-center space-x-4 mt-6">
        <button
          onClick={handlePrev}
          disabled={currentIndex === 0}
          className="bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-white px-8 py-3 rounded-lg hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors shadow-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
        >
          &larr; Prev
        </button>
        <button
          onClick={handleNext}
          disabled={currentIndex === currentCards.length - 1}
          className="bg-sky-600 text-white px-8 py-3 rounded-lg hover:bg-sky-700 transition-colors shadow-sm font-semibold disabled:bg-sky-400 disabled:cursor-not-allowed"
        >
          Next &rarr;
        </button>
      </div>
    </div>
  );
};

export default FlashcardsPage;