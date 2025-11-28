## Project Structure (preliminary) 

```bash
src/
├── app/
│   ├── page.tsx
│   ├── layout.tsx
│   └── globals.css
├── components/
│   └── AudioUploader.tsx
└── types/
    └── audio.ts
```


### Layer Explanation

### `app/` 
The Nextjs App router (each subfolder here with a page.tsx file will be a route) directory containing:
- **`page.tsx`** - Root page component that renders the landing page
- **`layout.tsx`** - Root layout wrapper that applies to all pages
- **`globals.css`** - Global CSS styles and Tailwind directives

### `components/` 
React components used throughout the application:
- **`AudioUploader.tsx`** - Main audio upload and player component
- Future components can be added here as the app grows

### `types/` 
TypeScript interfaces and type definitions 


## Setup


1. **Install dependencies**

```bash
npm install
```

2. **Run the development server**

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.


## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
