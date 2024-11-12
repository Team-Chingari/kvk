'use client'

import React, { useState } from 'react'

export default function MidiMelodyCompleter() {
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [completedMelody, setCompletedMelody] = useState<string | null>(null)

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0])
    }
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!file) return

    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Server response was not ok')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      setCompletedMelody(url)
    } catch (error) {
      console.error('Error:', error)
      alert('An error occurred while processing your file. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = () => {
    if (completedMelody) {
      const a = document.createElement('a')
      a.href = completedMelody
      a.download = 'completed-melody.mid'
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(completedMelody)
      document.body.removeChild(a)
    }
  }

  return (
    <div className="min-h-screen bg-[#FAF9F6] flex flex-col items-center justify-center p-4 text-gray-800 font-sans">
      <div className="flex items-center gap-4 mb-12">
        <div className="w-16 h-16 relative">
          <div className="absolute inset-0 bg-[#C85C3C] rounded-full transform rotate-45"></div>
          <div className="absolute inset-0 bg-[#C85C3C] rounded-full transform -rotate-45"></div>
          <div className="absolute inset-2 bg-[#FAF9F6] rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-[#C85C3C]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
          </div>
        </div>
        <span className="text-3xl font-semibold font-serif">MIDI Melody</span>
      </div>

      <h1 className="text-5xl font-serif mb-16 max-w-lg text-center leading-tight">
        Turn your melodies into complete compositions
      </h1>

      <div className="w-full max-w-md bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
        <p className="text-center mb-8 text-gray-600">
          Upload your MIDI file and let AI enhance your composition
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <button
            type="button"
            className="w-full h-14 relative border-2 hover:bg-gray-50 transition-colors flex items-center justify-center"
            onClick={() => document.getElementById('midi-file')?.click()}
          >
            <input
              id="midi-file"
              type="file"
              accept=".mid,.midi"
              onChange={handleFileChange}
              className="hidden"
            />
            <svg className="w-5 h-5 mr-2 text-[#C85C3C]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
            </svg>
            {file ? file.name : 'Choose MIDI file'}
          </button>

          <button 
            type="submit" 
            className={`w-full h-14 text-white transition-colors bg-[#C85C3C] hover:bg-[#B54D31] ${(!file || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={!file || isLoading}
          >
            {isLoading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </>
            ) : (
              'Enhance Melody'
            )}
          </button>
        </form>

        {completedMelody && (
          <div className="mt-6 text-center">
            <p className="text-green-600 mb-4">Your enhanced melody is ready!</p>
            <button 
              onClick={handleDownload}
              className="w-full h-14 border-2 hover:bg-gray-50 transition-colors flex items-center justify-center"
            >
              <svg className="w-5 h-5 mr-2 text-[#C85C3C]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
              </svg>
              Download Enhanced Melody
            </button>
          </div>
        )}

        <p className="text-xs text-center mt-6 text-gray-500">
          By continuing, you agree to our Terms of Service and Privacy Policy
        </p>
      </div>

      <button className="mt-8 text-gray-500 flex items-center gap-1">
        Learn more
        <span className="inline-block rotate-90">›</span>
      </button>

      <div className="fixed bottom-6 flex gap-2">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className={`w-2 h-2 rounded-full ${i === 0 ? "bg-gray-800" : "bg-gray-300"}`}
          />
        ))}
      </div>
    </div>
  )
}