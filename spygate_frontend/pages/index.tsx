import { useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to dashboard
    router.push('/dashboard');
  }, [router]);

  return (
    <>
      <Head>
        <title>SpygateAI - Pro Football Analysis</title>
        <meta name="description" content="AI-powered football analysis for competitive players" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="text-center">
          <div className="h-16 w-16 bg-spygate-orange rounded-lg flex items-center justify-center mx-auto mb-4">
            <span className="text-white font-bold text-xl">SG</span>
          </div>
          <h1 className="text-2xl font-bold text-dark-text mb-2">SpygateAI</h1>
          <p className="text-dark-text-muted">Loading...</p>
        </div>
      </div>
    </>
  );
}
