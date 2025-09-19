import React, { useState } from 'react';
import { QuizData, QuizQuestion, QuestionType } from '../types';
import CodeBlock from './CodeBlock';

interface QuizProps {
  quizData: QuizData;
}

const Quiz: React.FC<QuizProps> = ({ quizData }) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState<string[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [selectedOption, setSelectedOption] = useState<string | null>(null);
  const [isAnswered, setIsAnswered] = useState(false);

  const currentQuestion: QuizQuestion = quizData.questions[currentQuestionIndex];
  const score = userAnswers.reduce((acc, answer, index) => {
    return answer === quizData.questions[index].answer ? acc + 1 : acc;
  }, 0);

  const handleAnswer = (answer: string) => {
    if (isAnswered) return;
    setSelectedOption(answer);
    setIsAnswered(true);
    setUserAnswers([...userAnswers, answer]);
  };

  const handleNext = () => {
    if (currentQuestionIndex < quizData.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      resetQuestionState();
    } else {
      setShowResults(true);
    }
  };
  
  const resetQuestionState = () => {
    setSelectedOption(null);
    setIsAnswered(false);
  }

  const getOptionClass = (option: string) => {
    if (!isAnswered) {
      return 'bg-white dark:bg-slate-800 hover:bg-sky-50 dark:hover:bg-slate-700/50 border-slate-300 dark:border-slate-700';
    }
    if (option === currentQuestion.answer) {
      return 'bg-green-100 dark:bg-green-900/50 border-green-500 text-green-800 dark:text-green-300';
    }
    if (option === selectedOption && option !== currentQuestion.answer) {
      return 'bg-red-100 dark:bg-red-900/50 border-red-500 text-red-800 dark:text-red-300';
    }
    return 'bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-700 opacity-70';
  };

  if (showResults) {
    return (
      <div className="bg-white dark:bg-slate-800/50 p-8 rounded-lg shadow-lg text-center border border-slate-200 dark:border-slate-700 max-w-2xl mx-auto">
        <h2 className="text-3xl font-bold mb-4 text-slate-900 dark:text-white">Quiz Complete!</h2>
        <p className="text-xl mb-6">Your Score: <span className="font-bold text-sky-600 dark:text-sky-400">{score}</span> / {quizData.questions.length}</p>
        <button onClick={() => window.location.reload()} className="bg-sky-600 text-white px-6 py-3 rounded-lg hover:bg-sky-700 transition-colors font-semibold">
          Try a New Quiz
        </button>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-slate-800/50 p-6 sm:p-10 rounded-lg shadow-lg w-full max-w-3xl mx-auto border border-slate-200 dark:border-slate-700">
      <div className="mb-8">
        <p className="text-slate-500 dark:text-slate-400">Question {currentQuestionIndex + 1} of {quizData.questions.length}</p>
        <div className="mt-2 bg-slate-200 dark:bg-slate-700 rounded-full h-2.5">
          <div className="bg-sky-600 h-2.5 rounded-full" style={{ width: `${((currentQuestionIndex + 1) / quizData.questions.length) * 100}%` }}></div>
        </div>
      </div>

      <div className="my-8">
        {currentQuestion.type === QuestionType.CODE_OUTPUT ? (
            <div className="prose dark:prose-invert max-w-none text-justify">
                <p className="text-xl font-semibold mb-2 text-slate-900 dark:text-white">{currentQuestion.question.split('```')[0]}</p>
                <CodeBlock code={currentQuestion.question.split('```')[1]} language="python" />
            </div>
        ) : (
            <p className="text-xl font-semibold text-slate-900 dark:text-white text-justify">{currentQuestion.question}</p>
        )}
      </div>

      <div className="space-y-4">
        {currentQuestion.options?.map((option, index) => (
          <button
            key={index}
            onClick={() => handleAnswer(option)}
            disabled={isAnswered}
            className={`w-full text-left p-4 rounded-lg border-2 transition-all font-semibold ${getOptionClass(option)} ${isAnswered ? 'cursor-not-allowed' : 'cursor-pointer'}`}
          >
            {option}
          </button>
        ))}
      </div>
      
      {isAnswered && (
        <div className="mt-8 p-5 rounded-lg bg-sky-50 dark:bg-sky-900/20 border border-sky-200 dark:border-sky-900">
          <p className="font-bold text-slate-800 dark:text-white text-lg">{selectedOption === currentQuestion.answer ? "Correct!" : "Incorrect."}</p>
          <p className="mt-2 text-slate-600 dark:text-slate-300">{currentQuestion.explanation}</p>
          <button onClick={handleNext} className="mt-6 bg-sky-600 text-white px-6 py-3 rounded-lg hover:bg-sky-700 transition-colors w-full font-semibold">
            {currentQuestionIndex < quizData.questions.length - 1 ? 'Next Question' : 'Show Results'}
          </button>
        </div>
      )}
    </div>
  );
};

export default Quiz;
