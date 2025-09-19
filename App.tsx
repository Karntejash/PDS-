import React from 'react';
import { HashRouter, Routes, Route, NavLink } from 'react-router-dom';
import HomePage from './pages/HomePage';
import NotesPage from './pages/NotesPage';
import FlashcardsPage from './pages/FlashcardsPage';
import PYPPage from './pages/PYPPage';
import CheatSheetPage from './pages/CheatSheetPage';
import QuizGeneratorPage from './pages/QuizGeneratorPage';
import Header from './components/Header';
import Footer from './components/Footer';

export const parseMarkdownToHTML = (markdown: string): string => {
    if (!markdown) return '';
    
    // Process tables first as they are multi-line blocks
    const tableRegex = /(\|.*\|(?:\r\n|\n))\|[-| :]*\|(?:\r\n|\n)((?:\|.*\|(?:\r\n|\n)?)*)/g;
    let processedMarkdown = markdown.replace(tableRegex, (match, header, body) => {
        const headerRow = `<thead><tr>${header.split('|').slice(1, -1).map(h => `<th>${h.trim()}</th>`).join('')}</tr></thead>`;
        
        const bodyRows = body.trim().split(/\r\n|\n/).map(row => {
          const cells = row.split('|').slice(1, -1).map(cell => `<td>${cell.trim().replace(/\n/g, '<br/>')}</td>`).join('');
          return `<tr>${cells}</tr>`;
        }).join('');

        return `<div class="overflow-x-auto my-6 not-prose"><table class="w-full text-left border-collapse text-sm">${headerRow}<tbody>${bodyRows}</tbody></table></div>\n`;
    });

    const lines = processedMarkdown.split('\n');
    const htmlLines: string[] = [];
    let inList = false;
    let currentParagraph: string[] = [];

    const flushParagraph = () => {
        if (currentParagraph.length > 0) {
            const paragraphText = currentParagraph.join(' ').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            htmlLines.push(`<p>${paragraphText}</p>`);
            currentParagraph = [];
        }
    };

    for (const line of lines) {
        if (line.includes('<table')) {
            flushParagraph();
            htmlLines.push(line);
            continue;
        }

        const trimmedLine = line.trim();
        
        if (trimmedLine === '') { // Blank line signifies paragraph break
            flushParagraph();
            if (inList) {
                htmlLines.push('</ul>');
                inList = false;
            }
            continue; // Skip the blank line itself
        }

        if (trimmedLine.startsWith('* ') || trimmedLine.startsWith('- ')) {
            flushParagraph();
            if (!inList) {
                htmlLines.push('<ul>');
                inList = true;
            }
            htmlLines.push(`<li>${trimmedLine.substring(2).replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</li>`);
        } else {
            if (inList) {
                htmlLines.push('</ul>');
                inList = false;
            }

            if (trimmedLine.startsWith('# ')) {
                flushParagraph();
                htmlLines.push(`<h1>${trimmedLine.substring(2)}</h1>`);
            } else if (trimmedLine.startsWith('## ')) {
                flushParagraph();
                htmlLines.push(`<h2>${trimmedLine.substring(3)}</h2>`);
            } else if (trimmedLine.startsWith('### ')) {
                flushParagraph();
                htmlLines.push(`<h3>${trimmedLine.substring(4)}</h3>`);
            } else {
                currentParagraph.push(trimmedLine);
            }
        }
    }
    
    flushParagraph(); // Flush any remaining paragraph at the end
    if (inList) {
        htmlLines.push('</ul>');
    }

    return htmlLines.join('\n');
};


const App: React.FC = () => {
  return (
    <HashRouter>
      <div className="flex flex-col min-h-screen font-sans bg-white dark:bg-slate-900">
        <Header />
        <main className="flex-grow container mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-20">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/notes" element={<NotesPage />} />
            <Route path="/flashcards" element={<FlashcardsPage />} />
            <Route path="/pyp" element={<PYPPage />} />
            <Route path="/cheatsheet" element={<CheatSheetPage />} />
            <Route path="/quiz" element={<QuizGeneratorPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </HashRouter>
  );
};

export default App;
