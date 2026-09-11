import tseslint from 'typescript-eslint';
import hooks from 'eslint-plugin-react-hooks';
export default tseslint.config({ignores:['dist','src/schema.d.ts']}, ...tseslint.configs.recommended, {files:['src/**/*.{ts,tsx}'],plugins:{'react-hooks':hooks},rules:{...hooks.configs.recommended.rules}});
