import React from 'react';

interface FlashcardProps {
  question: string;
  answer: string;
  isFlipped: boolean;
  onFlip: () => void;
}

const Flashcard: React.FC<FlashcardProps> = ({ question, answer, isFlipped, onFlip }) => {
  return (
    <div
      className={`flashcard w-full h-80 perspective-1000 ${isFlipped ? 'flipped' : ''}`}
      onClick={onFlip}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onFlip()}
      aria-pressed={isFlipped}
    >
      <div className="flashcard-inner relative w-full h-full text-center">
        {/* Front of the card */}
        <div className="flashcard-front absolute w-full h-full bg-white dark:bg-slate-800 border-2 border-slate-200 dark:border-slate-700 rounded-lg shadow-md flex items-center justify-center p-6">
          <p className="text-xl md:text-2xl font-semibold text-slate-800 dark:text-slate-100">{question}</p>
        </div>
        {/* Back of the card */}
        <div className="flashcard-back absolute w-full h-full bg-sky-600 text-white rounded-lg shadow-md flex items-center justify-center p-6">
          <p className="text-lg md:text-xl">{answer}</p>
        </div>
      </div>
    </div>
  );
};

export default Flashcard;