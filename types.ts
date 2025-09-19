import React from 'react';

export interface Topic {
  title: string;
  content: string;
  code?: string;
  output?: string;
  diagram?: {
    component: React.ReactNode;
    caption: string;
  };
}

export interface Unit {
  id: number;
  title:string;
  topics: Topic[];
}

export interface Flashcard {
  id: number;
  question: string;
  answer: string;
}

export interface FlashcardDeck {
  id: string;
  title: string;
  description: string;
  cards: Flashcard[];
}


export interface PYPQuestion {
  id: string;
  question: string;
  answer: {
    theory?: string;
    code?: string;
    output?: string;
    diagram?: {
      component: React.ReactNode;
      caption: string;
    };
  };
}

export interface PYPaper {
  year: string;
  questions: PYPQuestion[];
}


export enum QuestionType {
    MCQ = 'MCQ',
    TRUE_FALSE = 'TRUE_FALSE',
    CODE_OUTPUT = 'CODE_OUTPUT',
}

export interface QuizQuestion {
    question: string;
    type: QuestionType;
    options?: string[];
    answer: string;
    explanation: string;
}

export interface QuizData {
    questions: QuizQuestion[];
}

export interface CheatSheetSection {
    title: string;
    // Fix: Import React to resolve React.ReactNode type
    content: React.ReactNode;
}