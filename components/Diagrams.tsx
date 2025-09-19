import React from 'react';

const DiagramBox: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className }) => (
  <div className={`text-center border-2 border-slate-400 dark:border-slate-500 rounded-md p-3 bg-white dark:bg-slate-800 text-sm font-semibold text-slate-700 dark:text-slate-200 ${className}`}>
    {children}
  </div>
);

const Arrow: React.FC = () => (
    <div className="flex items-center justify-center text-3xl font-light text-slate-400 dark:text-slate-500 mx-2">
        <span>→</span>
    </div>
);


export const DataSciencePipelineDiagram: React.FC = () => {
    const steps = ["Problem Definition & Data Acquisition", "Data Prep & Cleaning", "EDA", "Modeling", "Visualization & Communication", "Deployment"];
    return (
        <div className="flex flex-wrap items-center justify-center">
            {steps.map((step, index) => (
                <React.Fragment key={step}>
                    <DiagramBox>{step}</DiagramBox>
                    {index < steps.length - 1 && <Arrow />}
                </React.Fragment>
            ))}
        </div>
    );
};

export const DataScientistVennDiagram: React.FC = () => (
    <svg viewBox="0 0 300 180" className="w-full max-w-md" aria-label="Venn diagram showing the three core skills of a data scientist.">
        <circle cx="100" cy="80" r="60" fill="rgb(56 189 248 / 0.3)" stroke="rgb(14 116 144)" strokeWidth="2" />
        <circle cx="200" cy="80" r="60" fill="rgb(239 68 68 / 0.3)" stroke="rgb(153 27 27)" strokeWidth="2" />
        <circle cx="150" cy="120" r="60" fill="rgb(16 185 129 / 0.3)" stroke="rgb(6 95 70)" strokeWidth="2" />
        <text x="70" y="50" className="text-xs font-bold fill-current text-slate-800 dark:text-slate-100">Computer</text>
        <text x="78" y="62" className="text-xs font-bold fill-current text-slate-800 dark:text-slate-100">Science</text>
        <text x="185" y="50" className="text-xs font-bold fill-current text-slate-800 dark:text-slate-100">Math &</text>
        <text x="185" y="62" className="text-xs font-bold fill-current text-slate-800 dark:text-slate-100">Statistics</text>
        <text x="125" y="155" className="text-xs font-bold fill-current text-slate-800 dark:text-slate-100">Domain</text>
        <text x="123" y="167" className="text-xs font-bold fill-current text-slate-800 dark:text-slate-100">Expertise</text>
        <text x="135" y="95" className="text-sm font-extrabold fill-current text-slate-900 dark:text-white">Data</text>
        <text x="130" y="109" className="text-sm font-extrabold fill-current text-slate-900 dark:text-white">Science</text>
    </svg>
);

export const DataEcosystemDiagram: React.FC = () => (
    <div className="flex flex-col items-center">
        <div className="flex flex-wrap items-center justify-center">
            <DiagramBox>Big Data</DiagramBox>
            <Arrow/>
            <DiagramBox>Data Science<br/>(Process)</DiagramBox>
            <Arrow/>
            <DiagramBox>AI / ML<br/>(Product)</DiagramBox>
        </div>
        <div className="flex items-center justify-center mt-2 text-slate-500 dark:text-slate-400">
            <span className="text-3xl font-light">↺</span>
            <span className="ml-2 text-sm">Generates more data</span>
        </div>
    </div>
);

export const TrainTestSplitDiagram: React.FC = () => (
    <div className="flex flex-col items-center w-full max-w-lg">
        <DiagramBox className="w-full">Full Dataset</DiagramBox>
        <div className="w-px h-6 bg-slate-400 dark:bg-slate-500 my-1"></div>
        <div className="flex w-full">
            <div className="w-3/4 pr-2">
                 <DiagramBox className="w-full bg-sky-100 dark:bg-sky-900/50 border-sky-400 dark:border-sky-600">Training Set (~70-80%)</DiagramBox>
            </div>
            <div className="w-1/4 pl-2">
                 <DiagramBox className="w-full bg-green-100 dark:bg-green-900/50 border-green-400 dark:border-green-600">Testing Set (~20-30%)</DiagramBox>
            </div>
        </div>
         <p className="text-xs text-center mt-3 text-slate-500 dark:text-slate-400">Model learns from the Training Set and is evaluated on the unseen Testing Set.</p>
    </div>
);


const JoinDiagram: React.FC<{ title: string, shadedPath: string }> = ({ title, shadedPath }) => (
    <div className="flex flex-col items-center">
        <svg viewBox="0 0 100 60" className="w-28 h-20" aria-label={`Diagram illustrating a ${title} join.`}>
            <defs>
                <mask id={`mask-${title.toLowerCase().replace(' ', '-')}`}>
                    <rect x="0" y="0" width="100" height="60" fill="white" />
                    <circle cx="40" cy="30" r="20" fill="black" />
                    <circle cx="60" cy="30" r="20" fill="black" />
                </mask>
            </defs>
            <g className="fill-current text-sky-500/30 dark:text-sky-400/30">
                <path d={shadedPath} />
            </g>
            <circle cx="40" cy="30" r="20" fill="none" className="stroke-current text-slate-500 dark:text-slate-400" strokeWidth="1" />
            <circle cx="60" cy="30" r="20" fill="none" className="stroke-current text-slate-500 dark:text-slate-400" strokeWidth="1" />
            <text x="28" y="34" className="text-[8px] font-sans font-bold fill-current text-slate-700 dark:text-slate-200">A</text>
            <text x="68" y="34" className="text-[8px] font-sans font-bold fill-current text-slate-700 dark:text-slate-200">B</text>
        </svg>
        <p className="text-sm font-semibold mt-1 text-slate-700 dark:text-slate-200">{title}</p>
    </div>
);

export const SQLJoinsDiagram: React.FC = () => (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <JoinDiagram title="Inner Join" shadedPath="M 50 30 A 10 10 0 0 0 50 30 M 50,15 A 15 15, 0, 0, 1, 50 45 A 15 15, 0, 0, 1, 50 15 Z" />
        <JoinDiagram title="Left Join" shadedPath="M 40 30 A 20 20 0 1 0 40 30 Z" />
        <JoinDiagram title="Right Join" shadedPath="M 60 30 A 20 20 0 1 0 60 30 Z" />
        <JoinDiagram title="Outer Join" shadedPath="M 40 30 A 20 20 0 1 0 40 30 M 60 30 A 20 20 0 1 0 60 30 Z" />
    </div>
);