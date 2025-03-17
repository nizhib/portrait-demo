type Image = {
  url: string;
  width: number | null;
  height: number | null;
  alt: string | null;
};

export const sample = $state<Image>({
  url: '/images/sample1.webp',
  width: 597,
  height: 800,
  alt: 'Photo of Lomonosov',
});

export const image = $state(sample);
