import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { NAV_LINKS } from '../constants';
import ThemeToggle from './ThemeToggle';

const Header: React.FC = () => {
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const linkClasses = "px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200";
    const activeLinkClasses = "text-sky-600 dark:text-sky-400";
    const inactiveLinkClasses = "text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white";
    
    return (
        <header className="bg-white dark:bg-slate-900 sticky top-0 z-50 border-b border-slate-200 dark:border-slate-800">
            <nav className="container mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center">
                        <NavLink to="/" className="text-slate-900 dark:text-white font-bold text-xl">
                           PyDS Hub
                        </NavLink>
                    </div>
                    <div className="hidden md:block">
                        <div className="ml-10 flex items-center space-x-4">
                            <div className="flex items-baseline space-x-2">
                                {NAV_LINKS.map((link) => (
                                    <NavLink
                                        key={link.name}
                                        to={link.path}
                                        className={({ isActive }) =>
                                            `${linkClasses} ${isActive ? activeLinkClasses : inactiveLinkClasses}`
                                        }
                                    >
                                        {link.name}
                                    </NavLink>
                                ))}
                            </div>
                            <ThemeToggle />
                        </div>
                    </div>
                    <div className="-mr-2 flex items-center md:hidden">
                        <ThemeToggle />
                        <button
                            onClick={() => setIsMenuOpen(!isMenuOpen)}
                            type="button"
                            className="ml-2 inline-flex items-center justify-center p-2 rounded-md text-slate-400 hover:text-slate-500 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 focus:outline-none"
                            aria-controls="mobile-menu"
                            aria-expanded="false"
                        >
                            <span className="sr-only">Open main menu</span>
                            {isMenuOpen ? (
                                <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                            ) : (
                                <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" /></svg>
                            )}
                        </button>
                    </div>
                </div>
            </nav>

            {isMenuOpen && (
                <div className="md:hidden" id="mobile-menu">
                    <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                        {NAV_LINKS.map((link) => (
                            <NavLink
                                key={link.name}
                                to={link.path}
                                onClick={() => setIsMenuOpen(false)}
                                className={({ isActive }) =>
                                    `block ${linkClasses} ${isActive ? activeLinkClasses : inactiveLinkClasses}`
                                }
                            >
                                {link.name}
                            </NavLink>
                        ))}
                    </div>
                </div>
            )}
        </header>
    );
};

export default Header;