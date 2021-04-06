def findDecision(obj): #obj[0]: cap-shape, obj[1]: cap-surface, obj[2]: cap-color, obj[3]: bruises, obj[4]: odor, obj[5]: gill-attachment, obj[6]: gill-spacing, obj[7]: gill-size, obj[8]: gill-color, obj[9]: stalk-shape, obj[10]: stalk-root, obj[11]: stalk-surface-above-ring, obj[12]: stalk-surface-below-ring, obj[13]: stalk-color-above-ring, obj[14]: stalk-color-below-ring, obj[15]: veil-type, obj[16]: veil-color, obj[17]: ring-number, obj[18]: ring-type, obj[19]: spore-print-color, obj[20]: population, obj[21]: habitat
   # {"feature": "odor", "instances": 6499, "metric_value": 0.986, "depth": 1}
   if obj[4] == 'n':
      # {"feature": "stalk-shape", "instances": 3013, "metric_value": 0.2292, "depth": 2}
      if obj[9] == 't':
         return 'e'
      elif obj[9] == 'e':
         # {"feature": "spore-print-color", "instances": 517, "metric_value": 0.754, "depth": 3}
         if obj[19] == 'w':
            # {"feature": "stalk-surface-below-ring", "instances": 296, "metric_value": 0.5714, "depth": 4}
            if obj[12] == 's':
               # {"feature": "ring-type", "instances": 231, "metric_value": 0.2171, "depth": 5}
               if obj[18] == 'e':
                  return 'e'
               elif obj[18] == 'p':
                  # {"feature": "gill-size", "instances": 15, "metric_value": 0.9968, "depth": 6}
                  if obj[7] == 'n':
                     return 'p'
                  elif obj[7] == 'b':
                     return 'e'
                  else:
                     return 'e'
               else:
                  return 'p'
            elif obj[12] == 'y':
               return 'p'
            elif obj[12] == 'f':
               return 'e'
            elif obj[12] == 'k':
               return 'e'
            else:
               return 'e'
         elif obj[19] == 'r':
            return 'p'
         elif obj[19] == 'n':
            return 'e'
         elif obj[19] == 'k':
            return 'e'
         elif obj[19] == 'h':
            return 'e'
         elif obj[19] == 'o':
            return 'e'
         elif obj[19] == 'y':
            return 'e'
         else:
            return 'e'
      else:
         return 'e'
   elif obj[4] == 'f':
      return 'p'
   elif obj[4] == 'a':
      return 'e'
   elif obj[4] == 'l':
      return 'e'
   elif obj[4] == 'p':
      return 'p'
   elif obj[4] == 'y':
      return 'p'
   elif obj[4] == 's':
      return 'p'
   elif obj[4] == 'c':
      return 'p'
   elif obj[4] == 'm':
      return 'p'
   else:
      return 'p'
