import { GoogleGenAI, Type } from "@google/genai";
import { QuizData, Unit, PYPaper, PYPQuestion } from '../types';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
    throw new Error("API_KEY environment variable is not set.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

export const generateUnitSummary = async (unit: Unit): Promise<string> => {
    try {
        const unitContent = unit.topics.map(topic => `Topic: ${topic.title}\n${topic.content}`).join('\n\n');
        const prompt = `Please provide a concise markdown summary of the following unit content for a data science student. Focus on the key concepts, definitions, and Python library mentions. Use markdown for formatting, such as bolding key terms.

        Unit Title: ${unit.title}
        
        Content:
        ---
        ${unitContent}
        ---`;
        
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        return response.text;
    } catch (error) {
        console.error("Error generating unit summary:", error);
        return "Failed to generate summary. Please check the API key and try again.";
    }
};

export const generateQuestionSummary = async (question: PYPQuestion): Promise<string> => {
    try {
        const answerContent = `${question.answer.theory || ''}\n\n${question.answer.code || ''}`;
        const prompt = `Summarize the key concepts tested in the following exam question and its model answer. Explain the core idea in a way that's easy for a student to revise. Use markdown for formatting, with bolding for key terms and bullet points for lists.

        Question: ${question.question}

        Model Answer:
        ---
        ${answerContent}
        ---
        
        Concise Summary:`;
        
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        return response.text;
    } catch (error) {
        console.error("Error generating question summary:", error);
        return "Failed to generate summary. Please check the API key and try again.";
    }
};


export const predictExamQuestions = async (syllabus: Unit[], pyp: PYPaper[]): Promise<string> => {
    try {
        const syllabusText = syllabus.map(unit => `Unit ${unit.id}: ${unit.title}\n${unit.topics.map(t => `- ${t.title}`).join('\n')}`).join('\n\n');
        const pypText = pyp.map(paper => `Year: ${paper.year}\n${paper.questions.map(q => `- ${q.question}`).join('\n')}`).join('\n\n');

        const prompt = `Based on the provided syllabus and questions from previous year papers, predict 5 to 7 high-probability exam questions for the upcoming 'Python for Data Science' exam. The questions should cover a range of topics and reflect the patterns seen in past papers. Format the output as a numbered list.

        Syllabus:
        ---
        ${syllabusText}
        ---

        Previous Year Papers:
        ---
        ${pypText}
        ---

        Predicted Questions:
        `;

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        return response.text;
    } catch (error) {
        console.error("Error predicting exam questions:", error);
        return "Failed to predict questions. Please check the API key and try again.";
    }
};

export const generateQuiz = async (unit: Unit): Promise<QuizData> => {
    try {
        const unitContent = unit.topics.map(topic => `Topic: ${topic.title}\n${topic.content}\n${topic.code ?? ''}`).join('\n\n');
        const prompt = `Generate a 10-question quiz on the following content from the '${unit.title}' unit. The quiz should include a mix of question types: Multiple Choice (MCQ), True/False (TRUE_FALSE), and Code Output (CODE_OUTPUT). For CODE_OUTPUT questions, provide a Python code snippet and ask for the output. Provide a brief explanation for each answer.

        Unit Content:
        ---
        ${unitContent}
        ---
        `;

        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        questions: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    question: { type: Type.STRING },
                                    type: { type: Type.STRING, enum: ['MCQ', 'TRUE_FALSE', 'CODE_OUTPUT'] },
                                    options: { type: Type.ARRAY, items: { type: Type.STRING } },
                                    answer: { type: Type.STRING },
                                    explanation: { type: Type.STRING },
                                },
                            },
                        },
                    },
                },
            },
        });
        
        const jsonText = response.text.trim();
        const quizData = JSON.parse(jsonText);
        
        // Basic validation
        if (!quizData.questions || !Array.isArray(quizData.questions)) {
             throw new Error("Generated JSON is not in the expected format.");
        }
        
        return quizData as QuizData;

    } catch (error) {
        console.error("Error generating quiz:", error);
        throw new Error("Failed to generate quiz. The API may have returned an invalid format.");
    }
};

export const generateQuizFromPaper = async (paper: PYPaper): Promise<QuizData> => {
    try {
        const paperContent = paper.questions.map(q => `Question: ${q.question}\nAnswer Summary: ${q.answer.theory || q.answer.code}`).join('\n\n');
        const prompt = `You are a quiz generator for a 'Python for Data Science' course.
        Based on the questions and answers from the following past exam paper from the year ${paper.year}, generate a new 10-question quiz that tests the same core concepts. 
        Do not simply copy the questions. Instead, create new questions that assess a student's understanding of the underlying topics.
        The quiz must include a mix of question types: Multiple Choice (MCQ), True/False (TRUE_FALSE), and Code Output (CODE_OUTPUT).
        For CODE_OUTPUT questions, provide a Python code snippet and ask for the output.
        Provide a brief explanation for each answer.

        Past Paper Content:
        ---
        ${paperContent}
        ---
        `;

        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        questions: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.OBJECT,
                                properties: {
                                    question: { type: Type.STRING },
                                    type: { type: Type.STRING, enum: ['MCQ', 'TRUE_FALSE', 'CODE_OUTPUT'] },
                                    options: { type: Type.ARRAY, items: { type: Type.STRING } },
                                    answer: { type: Type.STRING },
                                    explanation: { type: Type.STRING },
                                },
                            },
                        },
                    },
                },
            },
        });
        
        const jsonText = response.text.trim();
        const quizData = JSON.parse(jsonText);
        
        if (!quizData.questions || !Array.isArray(quizData.questions)) {
             throw new Error("Generated JSON is not in the expected format.");
        }
        
        return quizData as QuizData;

    } catch (error) {
        console.error("Error generating quiz from paper:", error);
        throw new Error("Failed to generate quiz from paper. The API may have returned an invalid format.");
    }
};