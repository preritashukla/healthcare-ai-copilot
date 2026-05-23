import { useState, useEffect } from 'react'
import { WelcomeLanding } from './components/WelcomeLanding'
import './index.css'

export default function App() {
  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDark])

  return (
    <div className="h-screen w-full bg-theme-ambient text-theme-text-primary transition-colors duration-300 relative overflow-hidden overflow-y-auto">
      <WelcomeLanding isDark={isDark} toggleTheme={() => setIsDark(!isDark)} />
    </div>
  )
}
